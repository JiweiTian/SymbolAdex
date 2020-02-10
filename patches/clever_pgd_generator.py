import tensorflow as tf
import numpy as np
import string
import random 

import sys
sys.path.insert(0,'../cleverhans-master/')
from cleverhans.attacks import ProjectedGradientDescent
from cleverhans.model import CallableModelWrapper
from deepzono_milp import *

class CleverHansWrapperWrapper:
    def __init__( self, sess, in_tensor_name, out_tensor_name, mean, std ):
        self.sess = sess
        self.in_tensor_name = in_tensor_name
        self.out_tensor_name = out_tensor_name
        self.img_size = tf.get_default_graph().get_tensor_by_name( in_tensor_name ).shape[ 0 ].value
        self.mean = mean 
        self.std = std
    def __call__( self, img ):
        # Fix CleverHans's input tensor
        img = tf.reshape(img,(self.img_size,1)) # set size
        if not self.std is None:
            img = ( img - self.mean ) / self.std
        img = tf.cast(img, tf.float64) # convert to double to use in ERAN

        # Substitude their tensor in graph
        in_map = { self.in_tensor_name: img }
        return_elements = [ self.out_tensor_name ]
        output_tens = tf.import_graph_def( self.sess.graph.as_graph_def(),
                                           input_map=in_map,
                                           return_elements=return_elements )
        return output_tens[ 0 ]

class PGDGenerator:
    def __init__( self, sess, in_tensor_name, out_tensor_name, mean=None, std=None ):
        callable_model = CleverHansWrapperWrapper( sess, in_tensor_name, out_tensor_name, mean, std )
        cleverhans_model = CallableModelWrapper( callable_model, 'logits' )
        self.sess = sess
        self.attack = ProjectedGradientDescent( cleverhans_model, sess=sess )
        self.output_ten = tf.get_default_graph().get_tensor_by_name( out_tensor_name )
        self.input_ten = tf.get_default_graph().get_tensor_by_name( in_tensor_name )
        self.output_shape = [ dim.value for dim in self.output_ten.shape ]
        self.mean = mean
        self.std = std

    def generate( self, img, **kwargs ):
        if 'y_target' in kwargs:
            y_target = kwargs[ 'y_target' ]
            if not isinstance( y_target, (np.ndarray, list) ):
                y_target = [ y_target ]
            target_arr = np.zeros(self.output_shape)
            target = y_target[ np.random.randint(0, len(y_target)) ] 
            target_arr[ 0, target ] = 1
            kwargs[ 'y_target' ] = target_arr
        if 'eps_iter_size' in kwargs and not 'eps_iter' in kwargs and  'eps' in kwargs:
            kwargs['eps_iter'] = kwargs['eps'] * kwargs['eps_iter_size']
            del kwargs['eps_iter_size']
        check = True
        if 'check' in kwargs:
            check = kwargs['check']
            del kwargs['check'] 
        adv_example = self.attack.generate_np( img, **kwargs )
        if not self.std is None:
            adv_example = adv_example.reshape( -1, 1 )
            adv_example = ( adv_example - self.mean ) / self.std
            adv_example = adv_example.reshape( -1 )
        #assert ( np.all( adv_example <= kwargs['clip_max'] + 0.000001) and np.all( adv_example >= kwargs['clip_min'] - 0.000001 ) )
        real_cl = np.argmax( self.sess.run( self.output_ten, feed_dict={ self.input_ten: adv_example } ) )
        if not check:
            return adv_example

        if 'y_target' in kwargs and real_cl in y_target:
            return adv_example
        if not 'y_target' in kwargs:
            correct_cl = np.argmax( self.sess.run( self.output_ten, feed_dict={ self.input_ten: img } ) )
            if not correct_cl == real_cl:
                return adv_example

        return None

class NOctreeElement:

    def randomString( stringLength=10 ):
        """Generate a random string of fixed length """
        letters = string.ascii_lowercase
        return ''.join( random.choice( letters ) for i in range( stringLength ) )

    def __init__( self, *args, debug=False ):
        if not len( args ) == 2:
            raise ValueError( 'Wrong number of arguments' )
        if isinstance( args[1], NOctreeElement ): # Leaf
            self.child_idx = args[0]
            self.parent = args[1]
        else: # Root
            self.lb = args[0]
            self.ub = args[1]
            self.parent = None
            self.child_idx = 0
        
        self.dim = -1
        self.children = None
        self.debug = debug
        if debug:
            self.name = NOctreeElement.randomString()
    
    def split( self, dim ):
        if self.children is None:
            self.dim = dim
            self.children = ( NOctreeElement( 0, self, debug=self.debug ), NOctreeElement( 1, self, debug=self.debug ) )
        else:
            assert False, 'Already split.'
        return self.children
    
    def calculate_bounds( self ):
        parent = self
        idxs = []
        depth = 0
        while not( parent.parent is None ):
            idxs.append( parent.child_idx )
            parent = parent.parent
            depth += 1
        lb = parent.lb.copy()
        ub = parent.ub.copy()
        for i in range( len(idxs)-1, -1, -1 ):
            if idxs[i] == 0:
                ub[ parent.dim ] = ( lb[ parent.dim ] + ub[ parent.dim ] ) / 2.0
            else:
                lb[ parent.dim ] = ( lb[ parent.dim ] + ub[ parent.dim ] ) / 2.0 

            parent = parent.children[ idxs[ i ] ]

        return ( lb, ub, depth )

class PGDBoxAdex:
    def __init__( self, sess, in_tensor_name, out_tensor_name, **kwargs ):
        self.sess = sess
        self.input_ten = tf.get_default_graph().get_tensor_by_name( in_tensor_name )
        self.output_ten = tf.get_default_graph().get_tensor_by_name( out_tensor_name )


        self.generator = PGDGenerator( sess, in_tensor_name, out_tensor_name )
        if 'eps' in kwargs:
            self.eps = kwargs['eps']
            del kwargs['eps']
            if not isinstance( self.eps, (list, np.ndarray) ): # CleverHans doesn't like mixing number epsilons and array epsilon :(
                self.eps = np.array( [ self.eps ] * 784 )
            if isinstance( self.eps, list ):
                self.eps = np.array( self.eps )
            self.eps = self.eps.astype( 'float32' ) # CleverHans doesn't like float64 epsilons
        else:
            self.eps = None
        self.args = kwargs
        if 'y_target' in kwargs:
            assert False

    def get_class_box( self, img, y_target, eps, iters=100000 ):
        lb = np.array( [ np.inf ] * 784 )
        ub = np.array( [ -np.inf ] * 784 )
        bad_samples = 0 
        for i in range(iters):
            ex = self.generator.generate( img, y_target=y_target, eps=eps, **self.args)
            if not ex is None:
                cond = ex < lb
                not_cond = np.logical_not( cond ) 
                lb = np.select( [ cond, not_cond ] , [ ex, lb ] )
                cond = ex > ub
                not_cond = np.logical_not( cond ) 
                ub = np.select( [ cond, not_cond ] , [ ex, ub ] )
            else:
                bad_samples += 1
            if i % 1000 == 0:
                print( 'PGD bad attacks %i/%i' % ( bad_samples, i ) )
        print( 'PGD bad attacks %i/%i' % ( bad_samples, iters ) )
        if iters - bad_samples < 3:
            return ( None, None )
        return ( lb, ub )

    def visualise_class_box( self, img, y_target, eps, iters ):
        lb = np.array( [ np.inf ] * 784 )
        ub = np.array( [ -np.inf ] * 784 )

        bad_samples = 0
        exs = []
        for i in range(iters):
            ex = self.generator.generate( img, y_target=y_target, eps=eps, **self.args)
            if not ex is None:
                exs.append( ex )
                cond = ex < lb
                not_cond = np.logical_not( cond )
                lb = np.select( [ cond, not_cond ] , [ ex, lb ] )
                cond = ex > ub
                not_cond = np.logical_not( cond )
                ub = np.select( [ cond, not_cond ] , [ ex, ub ] )
            else:
                bad_samples += 1
            if i % 1000 == 0:
                print( 'PGD bad attacks %i/%i' % ( bad_samples, i ) )
        print( 'PGD bad attacks %i/%i' % ( bad_samples, iters ) )
        exs = np.array( exs )
        med = np.median(exs, axis = 0)
        mid = ( lb + ub ) / 2.0
        exs = ( exs - mid ) / ( lb - ub )
        exs = exs[ :, ub - lb > 0.0001 ]
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        fig.set_size_inches(60, 10)
        bp = ax.boxplot( exs.T )
        plt.setp(bp['boxes'], color='black')
        plt.setp(bp['whiskers'], color='black')
        plt.setp(bp['fliers'], color='red', marker='+')
        plt.savefig( 'pgd.png')
        plt.close(fig=fig)
        for i in range( exs.shape[1] ):
            print(i)
            fig, ax = plt.subplots()
            fig.set_size_inches(20, 10)
            plt.rcParams.update({'font.size': 22})
            plt.hist(exs[:,i], 100, density=True, facecolor='g' )
            plt.savefig( 'pgd_hist_%i.png' % i )
            plt.close(fig=fig)

    def get_class_box_med( self, img, y_target, eps, iters ):
        bad_samples = 0
        exs = []
        for i in range(iters):
            ex = self.generator.generate( img, y_target=y_target, eps=eps, **self.args)
            if not ex is None:
                exs.append( ex )
            else:
                bad_samples += 1
            if i % 1000 == 0:
                print( 'PGD bad attacks %i/%i' % ( bad_samples, i ) )
        print( 'PGD bad attacks %i/%i' % ( bad_samples, iters ) )
        exs = np.array( exs )
        med = np.median(exs, axis = 0)
        return med

    def generate_box_adex(  self, img, y_target, eran, domain, *args ):
        correct_cl = np.argmax( self.sess.run( self.output_ten, feed_dict={ self.input_ten: img } ) )

        #lb = np.clip( img - self.eps, self.args['clip_min'], self.args['clip_max'] )
        #ub = np.clip( img + self.eps, self.args['clip_min'], self.args['clip_max'] )
        #mid = ( lb + ub ) / 2.0
        #eps = ( ub - mid ).astype( 'float32' )
        #import pdb; pdb.set_trace()
        #lb, ub = self.get_class_box( mid, y_target, eps, iters=100 )
        lb, ub = self.get_class_box( img, y_target, self.eps, iters=50 )
        if lb is None: 
            return ( None, None ) 
        lb, ub = self.get_class_box( img, y_target, self.eps, iters=2000 )
        #self.visualise_class_box(  img, y_target, self.eps, iters=2000 )
        #lb = np.clip( img - self.eps, self.args['clip_min'], self.args['clip_max'] )
        #ub = np.clip( img + self.eps, self.args['clip_min'], self.args['clip_max'] )
        targets = [i for i in range(10) if not i == y_target]
        targets = correct_cl
        while True:
            mid = ( lb + ub ) / 2.0
            self.sample_attacks( lb, ub )
            eps_size = ( ub - mid ).astype( 'float32' ) # CleverHans doesn't like float64 epsilons
            lb_bad, ub_bad = self.get_class_box( mid, targets, eps_size, iters=1000 )
            if lb_bad is None:
                if not self.verify_network( lb, ub, y_target, eran, domain, *args ):
                    lb, ub = self.shrink_box( lb, ub, y_target, correct_cl, eran, domain, *args )
                    #self.refine_adex_with_octree( lb, ub, y_target, correct_cl, eran, domain, *args )
                    return ( lb, ub )
                else:
                    self.lb = lb
                    self.ub = ub
                    return ( lb, ub )
            lb_new, ub_new = self.diff_box( lb, ub, lb_bad, ub_bad )
            np.set_printoptions( precision=5, suppress=True )
            #print( 'Change:' ) 
            #print( np.abs( lb_new - lb ) + np.abs( ub_new - ub ) )
            #print( 'Size:' )
            #print( ub_new - lb_new )
            lb_dx = np.sum( np.abs( lb_new - lb ) )
            ub_dx = np.sum( np.abs( ub_new - ub ) )
            lb = lb_new 
            ub = ub_new
            print( 'Change:' , lb_dx + ub_dx )
            if lb_dx + ub_dx < 0.0001:
                if not self.verify_network( lb, ub, y_target, eran, domain, *args ):
                    lb, ub = self.shrink_box( lb, ub, y_target, correct_cl, eran, domain, *args )
                    return ( lb, ub )
                else:
                    self.lb = lb
                    self.ub = ub
                    return ( lb, ub )
        self.lb = lb
        self.ub = ub
        return ( lb, ub )

    def sample_attacks( self, lb, ub ):
        cnts = np.zeros(10)
        for i in range(10000):
            img = np.random.uniform( low=lb, high=ub )
            classification = np.argmax( self.sess.run( self.output_ten, feed_dict={ self.input_ten: img } ) )
            cnts[ classification ] += 1
        print( cnts )

    def shrink_box( self, lb, ub, y_target, correct_lab, eran, domain, *args ):
        mid = ( lb + ub ) / 2.0
        eps = ( ub - mid ).astype( 'float32' ) # CleverHans doesn't like float64 epsilons
        med = self.get_class_box_med( mid, y_target, eps, 1000 )

        eps = 1.0
        overall_eps = 0.0
        while eps > 0.00001:
            found = False
            while eps > 0.00001:
                eps /= 2.0
                lb_n = med + ( lb - med ) * ( overall_eps + eps )
                ub_n = med + ( ub - med ) * ( overall_eps + eps )
                if self.verify_network( lb_n, ub_n, y_target, eran, domain, *args ):
                    found = True
                    break
            if found:
                overall_eps += eps
                print ( 'Overall eps', overall_eps )
        if overall_eps < 0.00001:
            print( 'Failed to shrink box' )
            return ( None, None )
        print('Shrank with epsilon', overall_eps ) 
        return ( lb_n, ub_n )

    def check_pgd( self, lb, ub, y_target, iters=10 ):
        img = ( lb + ub ) / 2.0
        eps = ( ub - img ).astype( 'float32' ) # CleverHans doesn't like float64 epsilons
        img = np.clip( img, self.args['clip_min'], self.args['clip_max'] )
        bad_samples = 0
        for i in range(iters):
            ex = self.generator.generate( img, y_target=y_target, eps=eps, **self.args)
            if ex is None:
                bad_samples += 1
        return not iters == bad_samples

    def choose_split_for_octree( self, lb, ub, y_target, correct_lab ):
        self.grad = tf.gradients( self.output_ten[ 0, y_target ] - self.output_ten[ 0, correct_lab ], self.input_ten )
        grad = self.sess.run( self.grad, feed_dict={ self.input_ten: lb } )
        lbg = np.abs( grad )
        grad = self.sess.run( self.grad, feed_dict={ self.input_ten: ub } )
        ubg = np.abs( grad )
        dim = np.argmax( lbg + ubg )
        return dim
    
    def refine_adex_with_octree(  self, lb, ub, y_target, correct_lab, eran, domain, *args ):
        root = NOctreeElement( lb, ub, debug=True )
        els = [root]
        sols = []
        np.set_printoptions( precision=5, suppress=True )
        while( len( els ) > 0 ):
            el = els.pop(0)
            lb, ub, depth = el.calculate_bounds()
            print( depth )
            dim = self.choose_split_for_octree( lb, ub, y_target, correct_lab )
            l, r = el.split( dim )

            lb, ub, _ = l.calculate_bounds()

            if ( self.verify_network( lb, ub, y_target, eran, domain, *args ) ):
                print( 'Sol found' )
                sols.append( l )
            else:
                els.append( l )
            
            lb, ub, _ = r.calculate_bounds()
            if ( self.verify_network( lb, ub, y_target, eran, domain, *args ) ):
                print( 'Sol found' )
                sols.append( r )
            else:
                els.append( r )
        return root
 
    def sample_adexs( self ):
        import matplotlib.pyplot as plt
        for i in range(20):
            img = np.random.uniform(low=self.lb, high=self.ub)
            fig, ax = plt.subplots()
            im = ax.imshow( np.reshape(img, (28,28) ), vmin=0.0, vmax=1.0)
            fig.colorbar(im, ax=ax)
            plt.savefig( 'adv%i.png' % i )
        
    def diff_box( self, lb, ub, lb_bad, ub_bad ):
        lb_bad[ lb_bad > ub ] = ub[ lb_bad > ub ]
        lb_bad[ lb_bad < lb ] = lb[ lb_bad < lb ]
        ub_bad[ ub_bad < lb ] = lb[ ub_bad < lb ]
        ub_bad[ ub_bad > ub ] = ub[ ub_bad > ub ]
        lower_diff = lb_bad - lb
        upper_diff = ub - ub_bad
        cond = lower_diff > upper_diff
        not_cond = np.logical_not( cond )
        lb_res = np.select( [ cond, not_cond ], [ lb, ub_bad ] )
        ub_res = np.select( [ cond, not_cond ], [ lb_bad, ub ] )
        assert np.all( np.logical_or( lb_bad <= lb_res, lb_bad >= ub_res ) )
        assert np.all( np.logical_or( ub_bad <= lb_res, ub_bad >= ub_res ) )
        assert np.all( np.logical_and( lb <= lb_res, lb_res <= ub ) )
        assert np.all( np.logical_and( lb <= ub_res, ub_res <= ub ) )
        assert np.all( lb_res <= ub_res )
        return (lb_res, ub_res)

    def verify_network( self, lb, ub, y_target, eran, domain, *args ):
        perturbed_label, nn, nlb, nub = eran.analyze_box(lb, ub, domain, *args)
        if(perturbed_label==y_target):
            print("AdexFound")
            return True
        else:
            return False
            verified_flag,adv_image = verify_network_with_milp(nn, lb, ub, y_target, nlb, nub)
            if verified_flag == True:
                print('AdexFound with MILP' )
                return True
            else:
                print('Failed to verify' )
                return False
        
