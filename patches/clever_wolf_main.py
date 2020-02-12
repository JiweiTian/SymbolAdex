import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import gc
import time
import psutil
from multiprocessing import Process, Pipe
import numpy as np
import argparse
import csv

parser = argparse.ArgumentParser(description='ERAN Example',  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--netname', type=str, default=None, help='the network name, the extension can be only .pyt, .tf and .meta')
parser.add_argument('--domain', type=str, default='DeepPoly',choices=['LP', 'DeepPoly'], help='Domain to use in verification')
parser.add_argument('--dataset', type=str, default=None, help='the dataset, can be either mnist, cifar10, or acasxu')
parser.add_argument('--image_number', type=int, default=None, help='Whether to test a specific image.' )
parser.add_argument('--epsilon', type=float, default=0, help='the epsilon for L_infinity perturbation' )
parser.add_argument('--seed', type=int, default=None, help='Random seed for adex generation.' )
parser.add_argument('--model', type=str, default=None, help='Which model to load, if no model is specified a new one is trained.' )
parser.add_argument('--choose_criterea_every', type=int, default=10, help='How often to choose wheher to use LP or Wolfe' )
parser.add_argument('--max_cuts', type=int, default=50, help='Maximum number of cuts before shrinking' )
parser.add_argument('--save_every', type=int, default=10, help='How often to save model' )
parser.add_argument('--nowolf', action='store_true', help='Do not use Frank-Wolfe')
parser.add_argument('--obox_approx', action='store_true', help='Do not calculate full overapprox_box')

args = parser.parse_args()

if args.seed:
    seed = args.seed
    np.random.seed(seed)
else:
    seed = None
netname = args.netname
epsilon = args.epsilon
dataset = args.dataset

filename, file_extension = os.path.splitext(netname)
if file_extension not in [ '.pyt', '.tf' ]:
    raise argparse.ArgumentTypeError('only .pyt and .tf formats supported')
is_trained_with_pytorch = file_extension==".pyt"


def normalize(image, means, stds, is_conv):
    if(dataset=='mnist'):
        for i in range(len(image)):
            image[i] = (image[i] - means[0])/stds[0]
    elif(dataset=='mortgage'):
        image[ : ] = image[ : ] - means
        image[ : ] = image[ : ] / stds
    elif(dataset=='cifar10'):
        count = 0
        tmp = np.zeros(3072)
        for i in range(1024):
            tmp[count] = (image[count] - means[0])/stds[0]
            count = count + 1
            tmp[count] = (image[count] - means[1])/stds[1]
            count = count + 1
            tmp[count] = (image[count] - means[2])/stds[2]
            count = count + 1

        if(is_conv):
            for i in range(3072):
                image[i] = tmp[i]
        else:
            count = 0
            for i in range(1024):
                image[i] = tmp[count]
                count = count+1
                image[i+1024] = tmp[count]
                count = count+1
                image[i+2048] = tmp[count]
                count = count+1
    else:
        assert False

def denormalize(image, means, stds, is_conv):
    if(dataset=='mnist'):
        for i in range(len(image)):
            image[i] = image[i]*stds[0] + means[0]
    elif(dataset=='mortgage'):
        image[ : ] = image[ : ] * stds
        image[ : ] = image[ : ] + means
    elif(dataset=='cifar10'):
        count = 0
        tmp = np.zeros(3072)
        for i in range(1024):
            tmp[count] = image[count]*stds[0] + means[0]
            count = count + 1
            tmp[count] = image[count]*stds[1] + means[1]
            count = count + 1
            tmp[count] = image[count]*stds[2] + means[2]
            count = count + 1

        if(is_conv):
            for i in range(3072):
                image[i] = tmp[i]
        else:
            count = 0
            for i in range(1024):
                image[i] = tmp[count]
                count = count+1
                image[i+1024] = tmp[count]
                count = count+1
                image[i+2048] = tmp[count]
                count = count+1
    else:
        assert False
def create_pool( seed, netname, dataset, img, eps, clip_min, clip_max, model ):
    ncpus = os.sysconf("SC_NPROCESSORS_ONLN")
    conns = []
    procs = []
    parent_pid = os.getpid()
    for cpu in range( ncpus ):
        parent_conn, child_conn = Pipe()
        conns.append( parent_conn )
        threadseed = None
        if not seed is None:
            threadseed = seed + cpu
        p = Process(target=thread, args=( threadseed, netname, dataset, img, eps, clip_min, clip_max, child_conn, cpu, parent_pid, model ))
        p.start()
        procs.append( p )
    return conns, procs

def thread( seed, netname, dataset, im, eps, clip_min, clip_max, conn, proc_id, parent_pid, model ):
    import sys 
    # Prevent printing from child processes
    sys.stdout = open( str( proc_id ) + '.out', 'w')
    sys.stderr = open( str( proc_id ) + '.err', 'w')
    from clever_wolf import wolf_attack, create_PGD_gen
    import tensorflow as tf
    if not seed is None:
        tf.set_random_seed( seed )
    cut_model, is_conv, means, stds, _, pgd_means, pgd_stds, _ = create_tf_model( netname, dataset, im, model )
    args, pgd_gen = create_PGD_gen( cut_model, clip_min, clip_max, eps, pgd_means, pgd_stds )
    print( 'created PGD gen' )
    sys.stdout.flush()

    try:
        while True:
            if not conn.poll( 1 ):
                try:
                    process = psutil.Process(parent_pid)
                    print( 'Process status:', process.status() )
                    sys.stdout.flush()
                    continue
                except (psutil.ZombieProcess, psutil.AccessDenied, psutil.NoSuchProcess):
                    print( 'Parent dead' )
                    sys.stdout.flush()
                    return

            req, x0 = conn.recv()
            print( 'Req recieved:', req )
            if req == 'pgd':
                exs = []
                for i in range( x0[ 0 ] ):
                    ex = pgd_gen.generate( im, y_target=x0[ 1 ], **args )
                    exs.append( ex )
                print( 'pgd finished' ) 
                conn.send( exs )
            elif req == 'kill':
                break
            elif req == 'reset_model':
                cut_model.reset_model( *x0 )
                conn.send( 'done' )
            elif req == 'change_target':
                cut_model.update_target( *x0 )
                conn.send( 'done' )
            elif req == 'update_model':
                cut_model.update_bounds( *x0 )
                conn.send( 'done' )
            elif req == 'add_hyper':
                cut_model.add_hyperplane( *x0 )
                conn.send( 'done' )
            elif req == 'sample_around':
                ex = cut_model.wolf_sampling( x0, 200, 50 )
                conn.send( ex )
            elif req.startswith('neg_'):
                req = req[ 4: ]
                exs = []
                for init in x0:
                    ex = wolf_attack( cut_model.model, cut_model.xs, init, cut_model.tf_out_neg, cut_model.tf_grad_positive, cut_model.stopping_crit_negative, cut_model.tf_input, cut_model.sess, req )
                    exs.append( ex )
                print( 'wolf finished' )
                conn.send( exs )
            elif req.startswith('pos_'):
                req = req[ 4: ]
                exs = []
                for init in x0:
                    ex = wolf_attack( cut_model.model, cut_model.xs, init, cut_model.tf_out_pos, cut_model.tf_grad_negative, cut_model.stopping_crit_positive, cut_model.tf_input, cut_model.sess, req )
                    exs.append( ex )
                print( 'wolf finished' )
                conn.send( exs )
            else:
                assert False, 'Bad cmd'
            print( 'Done' )
            sys.stdout.flush()

    except Exception as e:
        import traceback
        traceback.print_exc(file=sys.stdout)
        print( e )
        sys.stdout.flush()

def map( data, op, print_every=100 ):
    data_size = len( data )
    mapping = {}
    for conn in conns:
        mapping[ conn ] = None
    sample_id = 0
    examples = [ None ] * data_size
    examples_gen = 0
    st = time.time()
    j = 0 
    while True:
        for conn in conns:
            if not mapping[ conn ] is None:
                status = conn.poll()
                if status:
                    examples_gen += 1
                    examples[ mapping[ conn ] ] = conn.recv()
                    mapping[ conn ] = None
            elif sample_id < data_size:
                mapping[ conn ] = sample_id
                conn.send( ( op, data[ sample_id ] ) )
                sample_id += 1
        if examples_gen == data_size:
            break
        else:
            if j * print_every < examples_gen:
                print( examples_gen, '/', data_size )
                j = int( examples_gen / print_every ) + 1
            time.sleep( 0.1 )
    end = time.time()
    print( end - st, 'sec' )
    return examples

def create_tf_model( netname, dataset, im, model_name ):
    import tensorflow as tf
    from read_net_file import read_tensorflow_net
    from clever_wolf import CutModel
    sess = tf.Session()
    filename, file_extension = os.path.splitext(netname)
    is_trained_with_pytorch = file_extension==".pyt"

    if(dataset=='mnist'):
        num_pixels = 784
    elif (dataset=='cifar10'):
        num_pixels = 3072
    elif(dataset=='acasxu'):
        num_pixels = 5
    elif(dataset=='mortgage'):
        num_pixels = 172
    model, is_conv, means, stds, layers = read_tensorflow_net(netname, num_pixels, is_trained_with_pytorch)
    pixel_size = np.array( [ 1.0 / 256.0 ] * num_pixels )
    pgd_means = np.zeros( ( num_pixels, 1 ) ) 
    pgd_stds = np.ones( ( num_pixels, 1 ) ) 
    if is_trained_with_pytorch:
        im_copy = np.copy( im )
        normalize( im_copy, means, stds, is_conv )
        if dataset == 'mnist':
            pgd_means[ : ] = means[ 0 ]
            pgd_stds[ : ] = stds[ 0 ]
            pixel_size = pixel_size / stds[ 0 ]
        elif dataset == 'cifar10': 
            if is_conv:
                count = 0 
                for i in range( 1024 ):
                    pixel_size[ count ] = pixel_size[ count ] / stds[ 0 ]
                    pgd_means[ count ] = means[ 0 ]
                    pgd_stds[ count ] = stds[ 0 ]
                    count = count + 1
                    pixel_size[ count ] = pixel_size[ count ] / stds[ 1 ]
                    pgd_means[ count ] = means[ 1 ]
                    pgd_stds[ count ] = stds[ 1 ]
                    count = count + 1
                    pixel_size[ count ] = pixel_size[ count ] / stds[ 2 ]
                    pgd_means[ count ] = means[ 2 ]
                    pgd_stds[ count ] = stds[ 2 ]
                    count = count + 1
            else:
                for i in range( 1024 ):
                    pixel_size[ i ] = pixel_size[ i ] / stds[ 0 ]
                    pgd_means[ i ] = means[ 0 ]
                    pgd_stds[ i ] = stds[ 0 ]
                    pixel_size[ i + 1024 ] = pixel_size[ i + 1024 ] / stds[ 1 ]
                    pgd_means[ i + 1024 ] = means[ 1 ]
                    pgd_stds[ i +1024 ] = stds[ 1 ]
                    pixel_size[ i + 2048 ] = pixel_size[ i + 2048 ] / stds[ 2 ]
                    pgd_means[ i + 2048 ] = means[ 2 ]
                    pgd_stds[ i + 2048 ] = stds[ 2 ]
        elif dataset == 'mortgage':
            pgd_means[ : , 0 ] = means
            pgd_stds[ : , 0 ] = stds
            pixel_size =  np.array( [ 1.0 ] * num_pixels ) / stds 
        else:
            # TODO Hack - works only on MNIST and CIFAR10 and mortgage
            assert False
    else:
        assert dataset == 'mnist'
        im_copy = np.copy( im )

    print( 'Model created' )
    tf_out = tf.get_default_graph().get_tensor_by_name( model.name )
    tf_in = tf.get_default_graph().get_tensor_by_name( 'x:0' )
    print( 'Tensors created' )

    out = sess.run( tf_out, feed_dict={ tf_in: im_copy } )
    print( 'Tf out computed' )
    if model_name is None:
        cut_model = CutModel( sess, tf_in, tf_out, np.argmax( out ), pixel_size )
    else:
        cut_model = CutModel.load( model_name, sess, tf_in, tf_out, np.argmax( out ) )
    print( 'Cut model created' )
    return cut_model, is_conv, means, stds, im_copy, pgd_means, pgd_stds, layers

def generate_initial_region_PGD( y_tar, num_samples ):
    arrs = map( [ ( 100, y_tar ) ] * int( num_samples / 100 ), 'pgd', print_every=5 )
    examples = []
    for arr in arrs:
        examples += filter( lambda x: not x is None, arr )
    examples = np.array( examples )
    print( examples.shape[ 0 ], '/', num_samples )
    
    if examples.shape[ 0 ] == 0:
        lb = None
        ub = None
    else:
        lb = np.min( examples, axis=0 )
        ub = np.max( examples, axis=0 )
    return ( examples, lb, ub )

def sample_wolf_attacks( inits, step_choice ):
    arrs = map( np.array_split( inits, int( inits.shape[ 0 ] / 10 ) ), step_choice )
    examples = []
    for arr in arrs:
        examples += arr
    attacks = []
    num_binary = 0
    k_sum = 0
    for attack, succ, binary, k in examples:
        if succ:
            k_sum += k
            if binary:
                num_binary += 1
            attacks.append( attack ) 
    attacks = np.array( attacks )
    avg_it = k_sum
    if not attacks.shape[ 0 ] == 0:
        avg_it /= attacks.shape[ 0 ]
    print( attacks.shape[ 0 ], '(', num_binary ,')/', inits.shape[ 0 ], 'Avg it:', avg_it  )

    return attacks

def sample_bad_attacks( inits ):
    if inits.shape[ 0 ] == 0:
        return inits
    arrs = map( inits, 'sample_around' )
    return np.concatenate( arrs )

def update_pool_barrier( cmd, params ):
    nthreads = len( conns )
    results = [ False ] * nthreads
    results = np.array( results ) 
    for conn in conns:
        conn.send( ( cmd, params ) )
    
    # Barrier
    while True:
        for i in range( nthreads ):
            msg = conns[ i ].poll()
            if msg:
                assert conns[ i ].recv() == 'done'
                results[ i ] = True
        if np.all( results ):
            break
        time.sleep( 0.1 )
        
def add_hyperplane_pool( params ):
    update_pool_barrier( 'add_hyper', params )

def reset_pool( params ):
    update_pool_barrier( 'reset_model', params )

def update_pool( params ):
    update_pool_barrier( 'update_model', params )

def update_target_pool( params ):
    update_pool_barrier( 'change_target', params )

def wolf_cut( cut_model, hist, num_samples, sample_poly=True ):
    res = False
    if sample_poly:
        samples = cut_model.sample_poly_under( num_samples )
    else:
        samples = cut_model.data
    print( 'Sampling done' )

    neg_ex = sample_wolf_attacks( samples, 'neg_binary_brent_before' )
    if neg_ex.shape[ 0 ] < samples.shape[ 0 ] / 10:
        print( 'Started sample_bad_attacks' )
        neg_ex = sample_bad_attacks( neg_ex )
        print( 'Finished sample_bad_attacks' )
    if neg_ex.shape[ 0 ] > 0:
        bad_idx, hyper = cut_plane( np.concatenate( ( neg_ex, *hist.cut_hist ) ), cut_model )
        bad_idx = bad_idx[ bad_idx < neg_ex.shape[ 0 ] ]
        if bad_idx.shape[ 0 ] < neg_ex.shape[ 0 ]:
            res = hyper
            cut_model.add_hyperplane( *hyper )
            add_hyperplane_pool( hyper )
            if hist.update_history( neg_ex ):
                print( [ hst.shape for hst in hist.cut_hist ] )

        if bad_idx.shape[ 0 ] > neg_ex.shape[ 0 ] * 0.8: 
            print( 'Started sample_bad_attacks' )
            neg_ex = sample_bad_attacks( neg_ex[ bad_idx ] )
            print( 'Finished sample_bad_attacks' )
            bad_idx, hyper = cut_plane( np.concatenate( ( neg_ex, *hist.cut_hist ) ), cut_model )
            bad_idx = bad_idx[ bad_idx < neg_ex.shape[ 0 ] ]
            if bad_idx.shape[ 0 ] < neg_ex.shape[ 0 ]:
                res = hyper
                cut_model.add_hyperplane( *hyper )
                add_hyperplane_pool( hyper )
                if hist.update_history( neg_ex ):
                    print( [ hst.shape for hst in hist.cut_hist ] )

    return res

def lp_cut( cut_model, hist, nn, domain, y_tar, lp_ver_output=None, complete=False ):
    if lp_ver_output is None:
        output = cut_model.lp_verification( nn, domain,  y_tar, complete=complete )
    else:
        output = lp_ver_output
    if isinstance( output, bool ):
        return True, None

    if domain == 'DeepPoly':
        eq, example, attack_class, bound = output
        print( 'Network output:', cut_model.eval_network( example ), 'LP output:', bound )
        example = ( eq, example )
    else:
        example, attack_class, bound = output
        in_example = example[ 0 : cut_model.input_size ]
        print( 'Network output:', cut_model.eval_network( in_example ), 'LP output:', bound )

    #hyper, sense = cut_model.lp_cut( nn, example, domain, y_tar, attack_class, nlb, nub )
    
    #new_data_idx = np.matmul( hyper[0], cut_model.data.T ) + hyper[ 1 ] < 0
    #new_data_idx = new_data_idx.reshape( -1 )
    #if sense == GRB.GREATER_EQUAL:
    #    new_data_idx = np.logical_not( new_data_idx )
    #cut_model.set_data( cut_model.data[ new_data_idx ] )
    st = time.time()
    samples = cut_model.lp_sampling( nn, example, 200, 5000, domain, y_tar, attack_class )
    dur = time.time() - st 
    print( 'Sampling:', dur, 'sec' )
    print( 'Verification cut',  samples.shape[ 0 ], '/', 5000 )

    bad_idx, hyper = cut_plane( np.concatenate( ( samples, *hist.cut_hist ) ), cut_model )
    bad_idx = bad_idx[ bad_idx < samples.shape[ 0 ] ]
    
    if bad_idx.shape[ 0 ] < samples.shape[ 0 ]:
        cut_model.add_hyperplane( *hyper )
        add_hyperplane_pool( hyper )
        if hist.update_history( samples ):
            print( [ hst.shape for hst in hist.cut_hist ] )

    return False, hyper

def choose_method( cut_model, hist, lp_params, wolf_params ):
    print( '\nStart choose method\n' )
    output = cut_model.lp_verification( *lp_params )
    
    if isinstance( output, bool ):
        return None

    bound = output[-1]

    model_wolf = cut_model.copy()
    hist_wolf = hist.copy()
    start = time.time()
    res = wolf_cut( model_wolf, hist_wolf, *wolf_params )
    if res == False:
        del model_wolf
        return 'LP'
    hyper_wolf = res
    time_wolf = time.time() - start
    
    model_lp = cut_model.copy()
    hist_lp = hist.copy()

    if len( output ) == 4:
        model_lp.set_precision( output[ 1 ] )
    else:
        model_lp.set_precision( output[ 0 ][ : cut_model.input_size ] )
    precision = model_lp.precision

    start = time.time()
    _, hyper_lp = lp_cut( model_lp, hist_lp, *lp_params, lp_ver_output=output )
    time_lp_sampl = time.time() - start

    output_wolf = model_wolf.lp_verification( *lp_params )
    cut_model.add_hyperplane( *hyper_wolf )
    add_hyperplane_pool( hyper_wolf )
    if isinstance( output_wolf, bool ):
        del model_lp, model_wolf
        return None
    
    if len( output ) == 4:
        in_example_wolf = output_wolf[ 1 ]
        bound_wolf = output_wolf[ -1 ]
    else:
        example_wolf, _, bound_wolf = output_wolf
        in_example_wolf = example_wolf[ 0 : cut_model.input_size ]

    start = time.time()
    output_lp = model_lp.lp_verification( *lp_params )
    time_ver = time.time() - start
    time_lp = time_ver + time_lp_sampl
    cut_model.add_hyperplane( *hyper_lp )
    add_hyperplane_pool( hyper_lp )
    if isinstance( output_lp, bool ):
        del model_lp, model_wolf
        return None
    if len( output ) == 4:
        in_example_lp = output_lp[ 1 ]
        bound_lp = output_lp[ -1 ]
    else:
        example_lp, _, bound_lp = output_lp
        in_example_lp = example_lp[ 0 : cut_model.input_size ]

    del model_lp, model_wolf

    dwolf = bound_wolf - bound
    dlp = bound_lp - bound

    if cut_model.check_if_inside( in_example_lp, precision=precision ).shape[ 0 ] == 0:
        lp_cut( cut_model, hist, *lp_params, lp_ver_output=output_lp )
    if cut_model.check_if_inside( in_example_wolf, precision=precision ).shape[ 0 ] == 0:
        lp_cut( cut_model, hist, *lp_params, lp_ver_output=output_wolf )

    print( '\nWolf_time:', time_wolf, 'LP time:', time_lp, 'Progress wolf:', dwolf, 'Progress lp:', dlp, '\n' )

    if ( dwolf / time_wolf ) > ( dlp / time_lp ):
        return 'Wolf'
    return 'LP'

class History:
    def __init__( self, input_size, cut_hist_size=5, update_hist_every=10 ):
        self.cut_hist_size = cut_hist_size
        self.update_hist_every = update_hist_every
        self.cut = 0
        self.cut_hist = []
        self.input_size = input_size
        for i in range( self.cut_hist_size ):
            self.cut_hist.append( np.zeros( ( 0, input_size ) ) )
                 
    def update_history( self, samples ):
        hist_is_full = [ not hist.shape[ 0 ] == 0 for hist in self.cut_hist ]
        hist_is_full = np.all( np.array( hist_is_full ) )
        
        updated = False
        if not hist_is_full or self.cut % self.update_hist_every == 0:
            updated = True
            for i in range( 1, self.cut_hist_size ):
                self.cut_hist[ i - 1 ] = self.cut_hist[ i ]
            if self.cut_hist_size > 0:
                self.cut_hist[ -1 ] = samples
        self.cut += 1

        return updated

    def copy( self ):
        copy = History( self.input_size, self.cut_hist_size, self.update_hist_every )
        copy.cut = self.cut
        copy.cut_hist = [ arr.copy() for arr in self.cut_hist ]

        return copy

def clever_wolf( nn, cut_model, y_true, y_tar, specLB, specUB, domain, args ):
    clever_start_time = time.time()
    if cut_model.y_tar == None:
        data, lb, ub = generate_initial_region_PGD( y_tar, 250 )
        reset_pool( ( specLB, specUB ) )                                                                                                                                
        update_target_pool( ( y_true, y_tar ) )                                                                                                                         
        cut_model.update_target( y_true, y_tar )
        cut_model.reset_model( specLB, specUB )
        succ_attacks = data.shape[ 0 ]
        all_attacks = 250
        if not args.nowolf:
            samples = cut_model.sample_poly_under( 250 )                                                                                                                   
            pos_ex = sample_wolf_attacks( samples, 'pos_brent' )
            succ_attacks += pos_ex.shape[ 0 ]
            all_attacks += 250
        print( 'Target', y_tar, succ_attacks, '/', all_attacks )
        if succ_attacks > 0:
            data, lb, ub = generate_initial_region_PGD( y_tar, 5000 )
            reset_pool( ( specLB, specUB ) )
            update_target_pool( ( y_true, y_tar ) )
            cut_model.update_target( y_true, y_tar )
            cut_model.reset_model( specLB, specUB )
            
            if not args.nowolf:
                samples = cut_model.sample_poly_under( 5000 )
                pos_ex = sample_wolf_attacks( samples, 'pos_brent' )
                if not pos_ex.shape[ 0 ] == 0:
                    data = np.concatenate( ( data, pos_ex ) ) 
            lb = np.min( data, axis=0 )
            ub = np.max( data, axis=0 )

            cut_model.update_bounds( lb, ub )
            cut_model.set_data( data )
            update_pool( ( lb, ub ) )
        else:
            return False
        s = time.time()
        config.dyn_krelu = False
        config.use_3relu = False
        config.use_2relu = False
        config.numproc_krelu = 24
        eran = ERAN(cut_model.tf_output, is_onnx=False)
        label,nn,nlb,nub = eran.analyze_box(lb, ub, 'deeppoly', 1, 1, True)
        print( 'Label:', label, 'Time:', time.time() - s, 's' )
        if label == -1:
            cut_model.nlb = [ np.array( lb ) for lb in nlb ]
            cut_model.nub = [ np.array( ub ) for ub in nub ]
            lb, ub = cut_model.overapprox_box()
            cut_model.nlb.insert( 0, lb )
            cut_model.nub.insert( 0, ub )
        else:
            cut_model.save( model_name )
            print( 'Verified, time:', int( time.time() - clever_start_time ) )
            return True   
    print( 'Init model' )
    if args.obox_approx:
        cut_model.approx_obox = True
    process = psutil.Process(os.getpid())
    start_lp_sampling = args.nowolf
    method = None
    res = None
    hist = History( cut_model.input_size, cut_hist_size=5, update_hist_every=2 ) 
    lp_params = ( nn, domain, y_tar )
    wolf_params = [ 1000 ]
    for cut in range( args.max_cuts ):

        sys.stdout.flush()

        print_vol( cut_model )

        gc.collect()
        print( '\nCut:', cut, ', Time:', int( time.time() - clever_start_time ), 'sec,', 'Target:', y_tar, ',X_size', cut_model.data_size, ',Memory:', process.memory_info().rss / (1024*1024),'\n')
        
        if cut % args.save_every == 0:
            cut_model.save( model_name )

        if not start_lp_sampling and cut % args.choose_criterea_every == 0:
            method = choose_method( cut_model, hist, lp_params, wolf_params )
            print( 'Chosen new method:', method )
            if method == None:
                cut_model.save( model_name )
                print( 'Verified, time:', int( time.time() - clever_start_time ) )
                return True

        if not start_lp_sampling and method == 'Wolf':
            res = wolf_cut( cut_model, hist, *wolf_params )

        if start_lp_sampling or res == False or method == 'LP':
            if not start_lp_sampling and not method == 'LP':
                start_lp_sampling = start_lp_sampling or res == False
            verified, _ = lp_cut( cut_model, hist, *lp_params )
            
            if verified:
                cut_model.save( model_name )
                print( 'Verified, time:', int( time.time() - clever_start_time ) )
                return True

    cut_model.shrink_poly( nn, domain, y_tar )
    cut_model.save( model_name )
    print_vol( cut_model )
    print( 'Verified, time:', int( time.time() - clever_start_time ) )
    return True


def destroy_pool():
    for conn in conns:
        conn.send( ( 'kill', None ) )
    for proc in procs:
        proc.join()


if(dataset=='mnist'):
    csvfile = open('../data/mnist_test_full.csv', 'r')
    tests = csv.reader(csvfile, delimiter=',')
elif(dataset=='cifar10'):
    csvfile = open('../data/cifar10_test.csv', 'r')
    tests = csv.reader(csvfile, delimiter=',')
elif(dataset=='mortgage'):
    csvfile = open('../data/mortgage_test.csv', 'r')
    tests = csv.reader(csvfile, delimiter=',')
elif(dataset=='acasxu'):
    specnumber = 9	
    specfile = '../data/acasxu/specs/acasxu_prop' + str(specnumber) +'_spec.txt'
    tests = open(specfile, 'r').read()
else:
    assert False
tests = list( tests )
test = tests[ args.image_number ]
corr_label = int( test[ 0 ] )

# Create img
if(dataset=='mnist'):
    image= np.float64(test[1:len(test)])/np.float64(255)
elif(dataset=='mortgage'):
    image= (np.float64(test[1:len(test)]))
elif(dataset=='cifar10'):
    if(is_trained_with_pytorch):
        image= (np.float64(test[1:len(test)])/np.float64(255))
    else:
        image= (np.float64(test[1:len(test)])/np.float64(255)) - 0.5

img = np.copy(image)

if dataset=='mortgage':
    feat_bounds = { 0: (0, 10), 5: (0, 200), 8: (1, 64), 9:(300, 850) }
    bounds_lb = [ feat_bounds[ key ][ 0 ] for key in feat_bounds ]
    bounds_ub = [ feat_bounds[ key ][ 1 ] for key in feat_bounds ]
    bounds_keys = list( feat_bounds.keys() )
    e = np.zeros( image.shape, dtype=np.float32 )
    e[ bounds_keys ] = epsilon
    clip_max = image.copy() 
    clip_max[ bounds_keys ] = bounds_ub
    clip_max += 1e-6
    clip_min = image.copy()
    clip_min[ bounds_keys ] = bounds_lb
    clip_min -= 1e-6
else:
    e = epsilon
    clip_max = np.array( [1.0 + 1e-6] * img.shape[ 0 ] ) # Fix clever hans num instabilities
    clip_min = np.array( [-1e-6] * img.shape[ 0 ] ) # Fix clever hans num instabilities

conns, procs = create_pool( seed, netname, dataset, img, e, clip_min, clip_max, args.model ) 
cut_model, is_conv, means, stds, img, _, _, layers = create_tf_model( netname, dataset, img, args.model )

import atexit
atexit.register(destroy_pool)
domain = args.domain
cut_model.create_tf_sampling_net( netname, is_trained_with_pytorch, domain )

'''
i = 0 
prec = 0
for test in tests:
    corr_label = int( test[ 0 ] )
    if(dataset=='mnist'):
        image= np.float64(test[1:len(test)])/np.float64(255)
    else:
        if(is_trained_with_pytorch):
            image= (np.float64(test[1:len(test)])/np.float64(255))
        else:
            image= (np.float64(test[1:len(test)])/np.float64(255)) - 0.5

    img = np.copy(image)
    if is_trained_with_pytorch:
        normalize( img, means, stds, is_conv )
    label = np.argmax( cut_model.sess.run( cut_model.tf_output, feed_dict={cut_model.tf_input: img } ) )
    i += 1
    if label == corr_label:
        prec += 1 
    if i % 100 == 0:
        print( prec , '/', i )
exit()
'''

from clever_wolf import *
import sys
sys.path.insert(0, '../ELINA/python_interface/')
from eran import ERAN
from config import config
config.dyn_krelu = False
config.numproc_krelu = 1
eran = ERAN(cut_model.tf_output, is_onnx=False)
imgLB = np.copy( img )
imgUB = np.copy( img )
label,nn,nlb,nub = eran.analyze_box(imgLB, imgUB, 'deepzono', 1, 1, True)
assert label == cut_model.y_true

# Create specLB/UB
if dataset=='mnist':
    specLB = np.clip(image - epsilon,0,1)
    specUB = np.clip(image + epsilon,0,1)
elif dataset=='mortgage':
    specLB = image.copy()
    specUB = image.copy()
    specLB[ bounds_keys ] = np.clip( image[ bounds_keys ] - epsilon, bounds_lb, bounds_ub )
    specUB[ bounds_keys ] = np.clip( image[ bounds_keys ] + epsilon, bounds_lb, bounds_ub )
elif dataset=='cifar10':
    if(is_trained_with_pytorch):
        specLB = np.clip(image - epsilon,0,1)
        specUB = np.clip(image + epsilon,0,1)
    else:
        specLB = np.clip(image-epsilon,-0.5,0.5)
        specUB = np.clip(image+epsilon,-0.5,0.5)
if is_trained_with_pytorch:
    normalize( specLB, means, stds, is_conv )
    normalize( specUB, means, stds, is_conv )

if not corr_label == label:
    print('Bad classification.')
    exit()

classes = cut_model.tf_output.shape.as_list()[1]

for i in range( classes ):
    if i == label:
        continue
    if not args.model is None:
        if not i == cut_model.y_tar:
            continue
    try:
        model_name = os.path.basename( filename ) + '_' + str( args.image_number ) + '_class_' + str( i )
        clever_wolf( nn, cut_model, corr_label, i, specLB, specUB, domain, args )
        if args.model is None:
            cut_model.y_tar = None
    except Exception as e:
        import traceback
        traceback.print_exc(file=sys.stdout)
        print(e)
        cut_model.save(model_name+'_error')

