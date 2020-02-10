import os
import time
import errno
import sys
import multiprocessing
from multiprocessing import Process, Manager
from gurobipy import *
import numpy as np
import tensorflow as tf
from sklearn.svm import LinearSVC
from scipy.optimize import minimize_scalar
from deepzono_milp import create_model
from clever_pgd_generator import PGDGenerator
from sklearn.linear_model import LogisticRegression

def brent( x_k, s_k, sess, tf_in, tf_out ):
    def f( y_k, x_k, s_k, sess, tf_inp, tf_out ):
        x_k_plus_one = x_k*( 1 - y_k ) + y_k*s_k
        out = sess.run( tf_out, feed_dict={ tf_in: x_k_plus_one } )
        return -out
    res = minimize_scalar( f, bounds=(0, 1), method='bounded', args=( x_k, s_k, sess, tf_in, tf_out ) )
    return res.x

def binary_search_step( x_k, s_k, max_yk, stopping_crit ):
    overall_yk = 0.0
    last = max_yk
    x_k_plus_1 = x_k*( 1 - ( last + overall_yk ) ) + ( last + overall_yk ) * s_k
    if not stopping_crit( x_k_plus_1 ):
        return max_yk
    while last >= 0.0000001:
        x_k_plus_1 = x_k*( 1 - ( last + overall_yk ) ) + ( last + overall_yk ) * s_k
        while last >= 0.0000001 and stopping_crit( x_k_plus_1 ):
            last /= 2.0
            x_k_plus_1 = x_k*( 1 - ( last + overall_yk ) ) + ( last + overall_yk ) * s_k
        overall_yk += last
        #import pdb; pdb.set_trace()
        if overall_yk > max_yk:
            assert False
    #print( overall_yk ) 
    return overall_yk + 0.00001

def wolf_attack( gurobi_model, gurobi_xs, x_0, tf_out, tf_grad, stopping_crit, tf_input, sess, step_choice ):
    x_k_printing = x_k = x_0
    print( step_choice )
    for k in range(1000):
        if k % 100 == 99:
            print( 'K', k + 1, 'val', sess.run( tf_out, feed_dict={tf_input: x_k} ), 'dist', np.sum( np.abs( x_k_printing - x_k ) ) )
            x_k_printing = x_k
        if stopping_crit( x_k ):
            #print( 'Found, K', k, 'val', sess.run( tf_out, feed_dict={tf_input: x_k} ) ) 
            return ( x_k, True, False, k )
        s_k = wolf_attack_step( gurobi_model, gurobi_xs, x_k, tf_grad, tf_input, sess )
        if step_choice == 'regular':
            y_k = 2.0 / (k + 2)
        elif step_choice == 'brent' or step_choice == 'binary_brent' or step_choice == 'binary_brent_before':
            y_k = brent( x_k, s_k, sess, tf_input, tf_out )
            if step_choice == 'binary_brent' or step_choice == 'binary_brent_before':
                x_k_plus_1 = x_k*( 1 - y_k ) + y_k*s_k
                if stopping_crit( x_k_plus_1 ):
                    y_k = binary_search_step( x_k, s_k, y_k, stopping_crit )
                    if step_choice == 'binary_brent_before':
                        y_k = y_k - 0.00001
                    x_k_plus_1 = x_k*( 1 - y_k ) + y_k*s_k
                    if step_choice == 'binary_brent_before':
                        return ( x_k_plus_1, True, True, k+1 )
                    if not stopping_crit( x_k_plus_1 ):
                        import pdb; pdb.set_trace()
        elif step_choice == 'binary':
            y_k = binary_search_step( x_k, s_k, 2.0 / (k + 2), stopping_crit )

        x_k_plus_1 = x_k*( 1 - y_k ) + y_k*s_k
        if ( np.sum( np.abs( x_k_plus_1 - x_k ) ) / x_k.shape[ 0 ] < 1e-6 ):
        #if ( np.sum( np.abs( x_k_plus_1 - x_k ) ) < 1e-6 ):
        #    print( 'Not found, K', k, 'val', sess.run( tf_out, feed_dict={tf_input: x_k} ) ) 
            return ( x_k, stopping_crit( x_k ), False, k+1 )
        x_k = x_k_plus_1
    #print( 'Not found, K', k, 'val', sess.run( tf_out, feed_dict={tf_input: x_k} ) ) 
    return ( x_k, False, False, k )

def wolf_attack_step( gurobi_model, gurobi_xs, x_k, tf_grad, tf_input, sess ):
    df = sess.run( tf_grad, feed_dict={ tf_input: x_k } )
    dim = len( gurobi_xs )
    obj = LinExpr()
    for i in range( dim ):
        obj += df[i] * gurobi_xs[i]
    gurobi_model.setObjective(obj,GRB.MINIMIZE)
    gurobi_model.optimize()
    if(gurobi_model.SolCount==0):
        assert False
    s_k = np.zeros( dim )
    for i in range( dim ):
        s_k[i] = gurobi_xs[i].x
    return s_k

def cut_plane( neg_ex, cut_model ):
    W, b = cut_model.fit_plane( cut_model.data, neg_ex )

    # Update dataset
    y_pred = np.matmul( W, cut_model.data.T ) + b > 0
    y_pred = y_pred.reshape( -1 )
    new_data = cut_model.data[ y_pred, : ]
    cut_model.set_data( new_data )
    
    y_pred = np.matmul( W, neg_ex.T ) + b < 0
    y_pred = y_pred.reshape( -1 )

    assert ( np.all( np.matmul( W, cut_model.data.T ) + b > 0 ) )

    bad_idx = np.where( np.logical_not( y_pred ) )[ 0 ]
    return ( bad_idx, ( W, b ) ) 

def print_vol( cut_model ):
    vol_under, vol_over = cut_model.calc_vol()
    print( 'Over:10^', vol_over )
    print( 'Under:10^', vol_under )

def verify_the_other_half( cut_model, model_rev, W, b, nn, y_tar ):

    input_size = cut_model.input_size
    constr_names = [ constr.ConstrName for constr in model_rev.model.getConstrs()]
    
    # Try verifying other half 
    model_rev.add_hyperplane( W, b, GRB.LESS_EQUAL )
    output = model_rev.lp_verification( nn, False,  y_tar )
    
    if isinstance( output, bool ):
        for constr in cut_model.model.getConstrs():
            if not constr.ConstrName in constr_names:
                return constr.ConstrName
    else:
        example, ver_model_rev, var_list_rev, bound = output
        in_example = example[ 0 : input_size ]
                
        del ver_model_rev
        del var_list_rev

        return in_example, bound

def create_PGD_gen( cut_model, clip_min, clip_max, eps, mean, std ):
    pgd_gen = PGDGenerator( cut_model.sess, cut_model.tf_input.name, cut_model.tf_output.name, mean, std )
    args = { 
            'eps': eps,
            'eps_iter_size': 0.1,
            'nb_iter': 50,
            'clip_min': clip_min,
            'clip_max': clip_max,
            }
    return args, pgd_gen

def pool_func_last_layer( var_name ):
    thread_model = global_model.copy()
    obj = LinExpr()
    obj += thread_model.getVarByName( global_target ) 
    obj -= thread_model.getVarByName( var_name )
    thread_model.reset()
    thread_model.setObjective(obj,GRB.MINIMIZE)
    thread_model.optimize()

    if thread_model.SolCount==0:
        assert False
    
    obj_val = thread_model.objbound

    bad_exam = []
    num_vars = len( global_var_list )
    for j in range( num_vars ) :
        var = thread_model.getVarByName( global_var_list[ j ] )
        bad_exam.append( var.x )
    bad_exam = np.array( bad_exam )

    del thread_model

    return obj_val, bad_exam

def pool_func_deeppoly( idx ):
    thread_model = global_model.copy()
    input_size = global_eq[ 0 ].shape[ 1 ]
    xs = [ thread_model.getVarByName( 'x' + str( i ) ) for i in range( input_size ) ]
    obj = LinExpr()
    for p in range( input_size ):
        obj += global_eq[ 0 ][ idx, p ] * xs[ p ]
    thread_model.reset()
    thread_model.setObjective( obj, GRB.MINIMIZE )
    thread_model.optimize()
    assert thread_model.SolCount == 1
    lb = thread_model.objbound + global_eq[ 2 ][ idx, 0 ]
    obj = LinExpr()
    for p in range( input_size ):
        obj += global_eq[ 1 ][ idx, p ] * xs[ p ]
    thread_model.reset()
    thread_model.setObjective( obj, GRB.MAXIMIZE )
    thread_model.optimize()
    assert thread_model.SolCount == 1
    ub = thread_model.objbound + global_eq[ 3 ][ idx, 0 ]
    
    if not global_get_example:
        return lb, ub

    bad_exam = []
    for p in range( input_size ) :
        bad_exam.append( xs[p].x )
    bad_exam = np.array( bad_exam )
    
    return lb, ub, bad_exam

def pool_func( var_name ):
    lb = -GRB.INFINITY
    ub = GRB.INFINITY
    thread_model = global_model.copy()
    obj = LinExpr()
    obj += thread_model.getVarByName( var_name )

    thread_model.setObjective(obj,GRB.MINIMIZE)
    thread_model.optimize()

    if not thread_model.SolCount==0:
        lb = thread_model.objbound
    else:
        assert False
    thread_model.reset()
    thread_model.setObjective(obj,GRB.MAXIMIZE)
    thread_model.optimize()
    if not thread_model.SolCount==0:
        ub = thread_model.objbound
    else:
        assert False
    
    del thread_model
    return lb, ub

class CutModel:
    def __init__( self, sess, tf_input, tf_output, y_true, pixel_size, y_tar=None, **kwargs ):
        self.sess = sess
        self.tf_input = tf_input
        self.input_size = tf_input.shape[ 0 ].value
        self.output_size = tf_output.shape[ -1 ].value
        self.pixel_size = pixel_size
        self.tf_output = tf_output
        
        self.tf_nlb = None
        self.tf_nub = None
        self.tf_attack = None
        self.tf_sampling_layers = None
        self.tf_sampling_x = None
        
        if not y_tar is None:
            self.update_target( y_true, y_tar )
        else:
            self.y_true = y_true
            self.y_tar = None

        if 'lb' in kwargs:
            lb = kwargs[ 'lb' ]
            ub = kwargs[ 'ub' ]
            self.reset_model( lb, ub )

        elif 'model' in kwargs:
            self.model = kwargs[ 'model' ]
            self.xs = kwargs[ 'xs' ]
            npdata = kwargs[ 'npdata' ]
            self.data = npdata[ 'data' ]
            if np.any( np.equal( self.data, None ) ):
                self.data = None
                self.data_size = 0
            else:
                self.data_size = self.data.shape[ 0 ]

            self.obox = npdata[ 'obox' ]
            if np.any( np.equal( self.obox, None ) ):
                self.obox = None

            self.ubox = npdata[ 'ubox' ]
            if np.any( np.equal( self.ubox, None ) ):
                self.ubox = None

            self.x0 = npdata[ 'x0' ]
            if np.any( np.equal( self.x0, None ) ):
                self.x0 = None

            self.precision = npdata[ 'precision' ]
            if np.any( np.equal( self.precision, None ) ):
                self.precision = None

            self.nlb = npdata[ 'nlb' ]
            self.nub = npdata[ 'nub' ]

            if type( self.nlb ) is np.ndarray:
                if self.nlb.size == 1 and self.nlb == None:
                    self.nlb = None
                self.nlb = self.nlb.tolist()
            if type( self.nub ) is np.ndarray:
                if self.nub.size == 1 and self.nub == None:
                    self.nub = None
                self.nub = self.nub.tolist()

            self.W = npdata[ 'W' ]
            self.model_nlb = npdata[ 'model_nlb' ]
            self.model_nub = npdata[ 'model_nub' ]
            self.cuts = npdata[ 'cuts' ]

            self.model.update()

    def set_data( self, data ):
        self.data = data
        self.data_size = data.shape[ 0 ]
    
    def reset_model( self, lb, ub ):
        model = Model( 'LP' )
        model.setParam( 'OutputFlag', 0 )
        
        xs = []
        input_size = self.input_size
        for i in range( input_size ):
            x = model.addVar(vtype=GRB.CONTINUOUS, lb=lb[i], ub=ub[i], name='x%i' % i)
            xs.append( x )

        self.model = model
        self.xs = xs
        self.cuts = 0
        self.invalidate_calc()
        self.nlb = self.nub = None
        self.W = np.zeros( ( 0, input_size + 1 ) )
        self.model_nlb = None
        self.model_nub = None
        self.update_W_bounds()

    def update_target( self, y_true, y_tar ):
        self.y_true = y_true
        self.y_tar = y_tar

        self.tf_out_pos = self.tf_output[ 0, y_tar ] - self.tf_output[ 0, self.y_true ]
        self.tf_grad_positive = tf.gradients( self.tf_out_pos, self.tf_input )[ 0 ]
        self.tf_out_neg = self.tf_output[ 0, self.y_true ] - self.tf_output[ 0, y_tar ]
        self.tf_grad_negative = tf.gradients( self.tf_out_neg, self.tf_input )[ 0 ]

        def stopping_crit_positive( x_k ):
            output = self.sess.run( self.tf_output, feed_dict={ self.tf_input: x_k } )
            return np.argmax( output ) == y_tar

        def stopping_crit_negative( x_k ):
            output = self.sess.run( self.tf_output, feed_dict={ self.tf_input: x_k } )
            return np.argmax( output ) == y_true

        self.stopping_crit_positive = stopping_crit_positive
        self.stopping_crit_negative = stopping_crit_negative
	
        self.data = None
        self.nlb = self.nub = None
        self.data_size = 0

    def invalidate_calc( self ):
        self.model.update()

        self.obox = None
        self.ubox = None
        self.x0 = None
        self.precision = None

    def update_bounds( self, lb, ub ):
        for i in range( self.input_size ):
            self.xs[ i ].setAttr( GRB.Attr.UB, ub[ i ] )
            self.xs[ i ].setAttr( GRB.Attr.LB, lb[ i ] )
        self.invalidate_calc()
        self.update_W_bounds()

    def update_W_bounds( self ):
        self.model_nlb = np.zeros( self.input_size )
        self.model_nub = np.zeros( self.input_size )
        for i in range( self.input_size ):
            self.model_nlb[ i ] = self.xs[ i ].LB
            self.model_nub[ i ] = self.xs[ i ].UB

    def add_W_constr( self, constr, assertOnEq=True ):
        line = []
        for x in self.xs:
            line.append( self.model.getCoeff( constr, x ) )
        line.append( -constr.RHS )
        line = np.array( line )
        if constr.Sense == '>':
            line *= -1.0
        if constr.Sense == '=':
            if assertOnEq:
                assert False
        self.W = np.concatenate( ( self.W, line.reshape( 1, -1 ) ) )
 
    @staticmethod
    def load( name, sess, tf_input, tf_output, y_true ): 
        npdata = np.load( name + '/npdata.npz', allow_pickle=True )
        
        model = read( name + '/' + name + '.lp' )
        model.setParam( 'OutputFlag', 0 )
        model.update()

        num_xs = len( model.getVars() )
        xs = [ model.getVarByName( 'x%i' % i ) for i in range( num_xs ) ]
        st0 = npdata[ 'rand_state' ]
        pixel_size = npdata[ 'pixel_size' ]
        st0 = tuple( st0.tolist() )
        np.random.set_state( st0 )
        y_tar = npdata[ 'y_tar' ]

        return CutModel( sess, tf_input, tf_output, y_true, pixel_size, y_tar=y_tar, model=model, xs=xs, npdata=npdata )
    
    def save( self, name ):
        name = name + '_it_' + str( self.cuts )
        try:
            os.makedirs( name )
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        self.model.write( name + '/' + name + '.lp' )
        rand_state = np.random.get_state()
        np.savez( name + '/npdata.npz', 
                  data=self.data, obox=self.obox, ubox=self.ubox, x0=self.x0, W=self.W, 
                  rand_state=rand_state, cuts=self.cuts, pixel_size=self.pixel_size, 
                  precision=self.precision, y_tar=self.y_tar, nlb=self.nlb, nub=self.nub, 
                  model_nlb=self.model_nlb, model_nub=self.model_nub )
        
    def copy( self ):
        model = self.model.copy()
        xs = [ model.getVarByName( 'x%i' % i ) for i in range( self.input_size ) ]
        
        if not self.data is None:
            data = self.data.copy()
        else:
            data = None

        if not self.x0 is None:
            x0 = self.x0.copy()
        else:
            x0 = None

        if not self.ubox is None:
            ubox = ( self.ubox[ 0 ].copy(), self.ubox[ 1 ].copy() )
        else:
            ubox = None

        if not self.obox is None:
            obox = ( self.obox[ 0 ].copy(), self.obox[ 1 ].copy() )
        else:
            obox = None

        if not self.W is None:
            W = self.W.copy()
        else:
            W = None
        
        if not self.W is None:
            model_nlb = self.model_nlb.copy()
        else:
            model_nlb = None

        if not self.model_nub is None:
            model_nub = self.model_nub.copy()
        else:
            model_nub = None

        npdata = { 'data': data, 'x0': x0, 'obox': obox, 'ubox': ubox, 
                   'W': W, 'cuts': self.cuts, 'pixel_size': self.pixel_size, 
                   'precision': self.precision, 'nlb': self.nlb,'nub': self.nub,
                   'model_nlb': model_nlb, 'model_nub': model_nub }

        copy = CutModel( self.sess, self.tf_input, self.tf_output, self.y_true, self.pixel_size, self.y_tar, model=model, xs=xs, npdata=npdata )
        
        copy.tf_nlb = self.tf_nlb
        copy.tf_nub = self.tf_nub
        copy.tf_attack = self.tf_attack
        copy.tf_sampling_layers = self.tf_sampling_layers
        copy.tf_sampling_x = self.tf_sampling_x
        copy.backsubstitute_tens = self.backsubstitute_tens

        return copy

    def add_hyperplane( self, W, b, Sense=GRB.GREATER_EQUAL ):
        constr = LinExpr()
        for i in range( self.input_size ):
            constr += W[ 0, i ] * self.xs[ i ]
        constr = self.model.addConstr( constr, Sense, -b )
        self.invalidate_calc()
        self.cuts += 1
        self.add_W_constr( constr )

    def sample_gaussian_around_attack( self, attack, directions, prints=True ):
        size = 1.0
        num_samples = directions.shape[ 0 ]
        best_samples = np.zeros( ( 0, directions.shape[ 1 ] ) )
        while size > 1e-9:
            s = time.time()
            samples = directions * size + attack
            good_idx = self.check_if_inside( samples )
            if good_idx.shape[ 0 ] > ( num_samples / 2.0 ):
                return samples[ good_idx ]
            if best_samples.shape[ 0 ] < good_idx.shape[ 0 ]:
                best_samples = samples[ good_idx ]
            size /= 2.0
            if prints:
                print( 'Sampl size:', size, 'Succ samples:', samples.shape[ 0 ], 'Time:', time.time()-s )
        return best_samples
    
    def set_precision( self, attack, precision_min=1e-6 ):
        precision = precision_min / 2.0
        while self.check_if_inside( attack, precision ).shape[ 0 ] == 0:
            precision *= 2
        precision *= 2

        self.precision = precision
        return self.precision

    def check_if_inside( self, samples, precision=None ):
        if precision == None:
            precision = self.precision
        
        if samples.ndim == 1:
            samples =samples.reshape( 1, -1 )
        num_samples = samples.shape[ 0 ]
        ones = np.ones( ( num_samples, 1 ) )
        samples_extended = np.concatenate( ( samples, ones ), axis=1 )
        matmul = np.matmul( self.W, samples_extended.T ) < precision
        full_idx = np.all( matmul, axis=0 )
        full_idx = np.logical_and( full_idx, np.all( ( samples - self.model_nlb ) > -precision, axis=1 ) )
        full_idx = np.logical_and( full_idx, np.all( ( samples - self.model_nub ) <  precision, axis=1 ) )
        good_idx = np.where( full_idx ) [ 0 ]
        return good_idx 

    def overapprox_box( self ):
        if not self.obox is None:
            return self.obox

        ncpus = os.sysconf("SC_NPROCESSORS_ONLN")

        global global_model
        global_model = self.model.copy()
        global_model.setParam(GRB.Param.Threads, 1)
        global_model.update()

        t = time.time()
        var_names = [ var.VarName for var in self.xs ]
        with multiprocessing.Pool(ncpus) as pool:
            solver_result = pool.map( pool_func, var_names )
        del globals()[ 'global_model' ]
        lb = np.array( [ 0 ] * self.input_size, dtype=np.float64 )
        ub = np.array( [ 0 ] * self.input_size, dtype=np.float64 )
        for i in range( self.input_size ):
            id = int( var_names[ i ][ 1 : ] )
            lb[ id ] = solver_result[ i ][ 0 ]
            ub[ id ] = solver_result[ i ][ 1 ]
        elapsed_time = time.time() - t
        print( 'Overapprox Time:', elapsed_time, 'secs' )

        self.obox = ( lb, ub )
        return self.obox
    
    def underapprox_box( self ):
        if not self.ubox is None:
            return self.ubox

        model_new = Model( 'Underbox' )
        model_new.setParam( 'OutputFlag', 0 )
        vars_new = {}

        lbo, ubo = self.overapprox_box()
        for i in range( self.input_size ):
            var_lo = model_new.addVar(vtype=GRB.CONTINUOUS, lb=lbo[ i ], ub=ubo[ i ], name='lo_' + self.xs[ i ].VarName)
            var_hi = model_new.addVar(vtype=GRB.CONTINUOUS, lb=lbo[ i ], ub=ubo[ i ], name='hi_' + self.xs[ i ].VarName)
            vars_new[ self.xs[ i ] ] = ( var_lo, var_hi )
            constr_new = LinExpr()
            constr_new += var_hi - var_lo
            model_new.addConstr( constr_new, GRB.GREATER_EQUAL, 0 )

        for constr in self.model.getConstrs():
            constr_new = LinExpr()
            for x in self.xs:
                coef = self.model.getCoeff( constr, x )
                if ( coef >= 0 and constr.Sense == '<' ) or ( coef < 0 and constr.Sense == '>' ) :
                    constr_new += coef*vars_new[ x ][ 1 ]
                if ( coef < 0 and constr.Sense == '<' ) or ( coef >= 0 and constr.Sense == '>' ) :
                    constr_new += coef*vars_new[ x ][ 0 ]
                if constr.Sense == '=':
                    assert False
            model_new.addConstr( constr_new, constr.Sense, constr.RHS )
        
        obj = LinExpr()
        for x in self.xs:
            obj += vars_new[ x ][ 1 ] - vars_new[ x ][ 0 ]
        model_new.setObjective( obj, GRB.MAXIMIZE )
        model_new.optimize()
        if model_new.SolCount == 0:
            assert False

        lb = np.zeros( self.input_size )
        ub = np.zeros( self.input_size )
        for i in range( self.input_size ):
            x = self.xs[ i ]
            lb[ i ] = vars_new[ x ][ 0 ].x
            ub[ i ] = vars_new[ x ][ 1 ].x

        del model_new
        
        self.ubox = ( lb, ub )
        return self.ubox

    def get_x0( self ):
        if not self.x0 is None:
            return self.x0

        model = Model( 'InitPoint' )
        model.setParam( 'OutputFlag', 0 )
        eps = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY, name='eps')
        for var in self.model.getVars():
            if not var.LB is -GRB.INFINITY and not var.UB is GRB.INFINITY and var.UB - var.LB < 1e-5:
                var_copy = model.addVar(vtype=GRB.CONTINUOUS, lb=var.LB, ub=var.UB, name=var.VarName)
                continue
            var_copy = model.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name=var.VarName)
            if not var.LB is -GRB.INFINITY:
                constr = LinExpr()
                constr += var_copy - eps
                model.addConstr(constr, GRB.GREATER_EQUAL, var.LB)
            if not var.UB is GRB.INFINITY:
                constr = LinExpr()
                constr += var_copy + eps
                model.addConstr(constr, GRB.LESS_EQUAL, var.UB)
        model.update()

        for constr in self.model.getConstrs():
            constr_new = LinExpr()
            coefs = []
            for var in self.model.getVars():
                var_new = model.getVarByName( var.VarName )
                coef = self.model.getCoeff( constr, var )
                constr_new += coef * var_new
                coefs.append( coef )
            norm = np.linalg.norm( coefs )
            if constr.Sense == '<':
                constr_new += norm * eps
                model.addConstr(constr_new, GRB.LESS_EQUAL, constr.RHS)
            elif constr.Sense == '>':
                constr_new -= norm * eps
                model.addConstr(constr_new, GRB.GREATER_EQUAL, constr.RHS)
            else:
                assert False

        obj = LinExpr()
        obj += eps
        model.setObjective(obj,GRB.MAXIMIZE)
        model.optimize()
        if model.SolCount == 0:
            assert False

        res = np.zeros( self.input_size )
        for i in range( self.input_size ):
            var = model.getVarByName( self.xs[ i ].VarName )
            res[ i ] = var.x
        print ( 'X0_eps: ', model.getVarByName( 'eps' ).x )
        del model

        self.x0 = res
        return self.x0

    def sample_poly_under( self, num_samples ):
        '''Underapprox box and sample from it'''
        lb, ub = self.underapprox_box()
        samples = np.random.uniform( low=lb, high=ub, size=( num_samples, lb.shape[ 0 ] ) )
        return samples

    def sample_poly_mcmc( self, burn_in, num_samples, skip ):
        """ Implementation of hit-and-run sampling """
        dim = self.input_size
        x_k = self.get_x0()
        samples = np.zeros( ( int( num_samples / ( skip + 1 ) ), dim ) )
        lbo, ubo = self.overapprox_box()
        j = 0

        def sample_direction( dim ):
            unnorm_dir = np.random.normal( 0, 1, size=( dim ) )
            l2_norm = np.linalg.norm( unnorm_dir, ord=2 )
            direct = unnorm_dir / l2_norm
            return direct

        while j <  burn_in + num_samples:
            direct = sample_direction( dim )
            direct[ np.abs( ubo - lbo ) < 1e-5 ] = 0
            model = self.model.copy()
            theta = model.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name='theta')

            for i in range( dim ):
                constr = LinExpr()
                constr += model.getVarByName( self.xs[ i ].VarName )
                constr -= theta * direct[i]
                model.addConstr( constr, GRB.EQUAL, x_k[ i ] )

            obj = LinExpr()
            obj += theta
            model.setObjective(obj,GRB.MINIMIZE)
            model.optimize()
            if(model.SolCount==0):
                del model
                print( 'Bad dir' )
                continue
            #else:
            #    print( 'Good dir' )
            min = model.objval

            model.setObjective(obj,GRB.MAXIMIZE)
            model.optimize()
            if(model.SolCount==0):
                del model
                print( 'Bad dir' )
                continue
            #else:
            #    print( 'Good dir' )
            max = model.objval
            if min > max:
                del model
                print( 'Bad dir max < min' )
                continue
            #print ( min, ':', max )
            x_k = x_k + direct * np.random.uniform( low=min, high=max )
            if j >= burn_in:
                if (j-burn_in) % ( skip + 1 ) == 0:
                    idx = ( j - burn_in ) / ( skip + 1 )
                    idx = int( idx )
                    samples[ idx, : ] = x_k
                if j % 100 == 99:
                    print( 'Generated', j-burn_in )
            else:
                if j % 100 == 99:
                    print( 'Burn-in', j )
            j += 1
            del model


        return samples
    
    def calc_vol( self ):
        lbo, ubo = self.overapprox_box()
        lbu, ubu = self.underapprox_box()

        sizes = ( ( ubo - lbo ) / self.pixel_size ).astype( np.int64 )
        sizes = sizes[ sizes > 0 ]
        vol_over = np.sum( np.log10( sizes ) )

        sizes = ( ( ubu - lbu ) / self.pixel_size ).astype( np.int64 )
        sizes = sizes[ sizes > 0 ]
        vol_under = np.sum( np.log10( sizes ) )

        return ( vol_under, vol_over )

    def eval_network( self, input ):
        out = self.sess.run( self.tf_out_pos, feed_dict={ self.tf_input: input } )
        return out

    def binary_line_search_for_sampling( self, x0, attack, accept ):
        ministep = 1.0
        step = 0.0
        while ministep > 1e-7:
            while ministep > 1e-7:
                ministep /= 2.0
                point = ( step + ministep ) * x0 + ( 1.0 - step - ministep ) * attack
                if accept( point ):
                    break
            step += ministep
        print( 'Sampling step:', step )
        point = step * x0 + ( 1.0 - step ) * attack
        return point
  
    def sampling_around_negative_example( self, attack, filter, num_samples_est, num_samples, x0=None ):
        print( 'Entered sampling' )

        lb, ub = self.overapprox_box()
        rand_dir = np.random.normal( 0, 1, size=( num_samples_est, self.input_size ) ) 
        rand_dir *= ( ub - lb )
        if x0 is None:
            x0_new = self.get_x0()
        
        self.set_precision( attack )
        def accept( x_k ):
            samples = self.sample_gaussian_around_attack( x_k, rand_dir, prints=False )
            samples = filter( samples )
            return samples.shape[ 0 ] > num_samples_est / 4.0
        
        s = time.time()
        print( 'Before binary' )
        if x0 is None:
            pt = self.binary_line_search_for_sampling( x0_new, attack, accept )
        else:
            pt = x0
        print( 'After binary:', time.time() - s, 'sec'  )

        rand_dir = np.random.normal( 0, 1, size=( int( num_samples / 10.0 ), self.input_size ) ) 
        rand_dir *= ( ub - lb )
        samples_final = []
        for coef in np.arange( 0, 1, 0.1 ):
            x_k = pt * coef + attack * ( 1 - coef )
            samples = self.sample_gaussian_around_attack( x_k, rand_dir, prints=False )
            samples = filter( samples )
            samples_final.append( samples )
        samples = np.concatenate( samples_final, axis=0 )
        return samples

    def lp_cut( self, nn, attack, ver_type, target, attack_class ):
        lp_ver = LP_verifier( attack, self.model, self.xs, nn, self.nlb, self.nub, ver_type )
        in_attack = attack[ : self.input_size ]
        x0 = self.get_x0()
        x0 = lp_ver.get_x0( x0, target, attack_class )
        W = in_attack - x0
        W = W.reshape( 1, -1 )
        b = np.dot( W, x0 )
        if np.matmul( W, in_attack ) + b < 0:
            sense = GRB.GREATER_EQUAL
        else:
            sense = GRB.LESS_EQUAL
        return ( W, b ), sense

    def lp_sampling( self, nn, attack, num_samples_est, num_samples, ver_type, target, attack_class ):
        #lp_ver = LP_verifier( attack, self.model, self.xs, nn, self.nlb, self.nub, ver_type )
        if not ver_type == 'DeepPoly':
            in_attack = attack[ : self.input_size ]
        else:
            in_attack = attack[1]
        #x0 = lp_ver.get_x0( x0, target, attack_class )

        def filter( samples ):
            if ver_type == 'DeepPoly':
                idx = np.matmul( samples, attack[ 0 ][ 0 ] ) +  attack[ 0 ][ 1 ] < 0
                return samples[ idx ]
            return self.tf_lp_sampling( attack, samples, ver_type )
        return self.sampling_around_negative_example( in_attack, filter, num_samples_est, num_samples )

    def wolf_sampling( self, attack, num_samples_est, num_samples ):
        def filter( samples ):
            idx = []
            for i in range( samples.shape[ 0 ] ):
                if self.stopping_crit_negative( samples[ i ] ):
                    idx.append( i )
            return samples[ idx ]
        return self.sampling_around_negative_example( attack, filter, num_samples_est, num_samples )

    def lp_verification( self, nn, ver_type, target, complete=False ):
        if ver_type == 'MILP':
            use_milp = True
        else:
            use_milp = False
        if ver_type == 'DeepPoly':
            return self.extract_deeppoly_backsub( target )
            use_deeppoly = True
        else:
            use_deeppoly = False
        
        ncpus = os.sysconf("SC_NPROCESSORS_ONLN")
        self.full_models=[]
        input_size = self.input_size
        
        if self.nlb is None:
            self.nlb = [ np.array( [] ) ] * ( nn.numlayer + 1 )
        if self.nub is None:
            self.nub = [ np.array( [] ) ] * ( nn.numlayer + 1 )

        if complete or len( self.nlb[ 0 ] ) == 0 or len( self.nub[ 0 ] ) == 0:
            LB_N0, UB_N0 = self.overapprox_box()
            self.nlb[ 0 ] = LB_N0
            self.nub[ 0 ] = UB_N0
        for layerno in range( nn.numlayer ):
            relu_needed = [0]*(layerno+1)
            for i in range(layerno):
                relu_needed[i] = 1
            krelu_groups = [ None ]*(layerno+1)
            if use_deeppoly:
                deeppoly = [1]*(layerno+1)
            else:
                deeppoly = False

            print( 'Layer:', layerno+1, 'MILP:', use_milp, 'DeepPoly:', deeppoly )
            counter, var_list, model = create_model(nn, self.nlb[ 0 ], self.nub[ 0 ], self.nlb[ 1 : layerno + 1 ] + [[]], self.nub[ 1 : layerno + 1 ] + [[]], krelu_groups, layerno + 1, use_milp, relu_needed, deeppoly)

            num_vars = len( var_list )
            output_size = num_vars - counter
            model.update()
            for constr in self.model.getConstrs():
                constr_new = LinExpr()
                for i in range( input_size ):
                    coef = self.model.getCoeff( constr, self.xs[ i ] )
                    constr_new += coef * var_list[ i ]
                    assert var_list[i].VarName == self.xs[i].VarName
                model.addConstr( constr_new, constr.Sense, constr.RHS )
            
            self.full_models.append(model)

            global global_model
            global_model = model

            if layerno == nn.numlayer - 1:
                model.setParam(GRB.Param.Threads, 1)
                model.update()

                t = time.time()

                global global_var_list
                global_var_list = [ var.VarName for var in var_list ]
                
                global global_target
                global_target = var_list[ counter + target ].VarName

                adv_targets = [ var.VarName for var in var_list[ counter : ] if not global_target == var.VarName ] 
                with multiprocessing.Pool(ncpus) as pool:
                    solver_result = pool.map( pool_func_last_layer, adv_targets )
                
                del globals()[ 'global_model' ]
                del globals()[ 'global_var_list' ]
                del globals()[ 'global_target' ]

                smallestobj = 0.0
                bad_exam = None
                bad_class = None
                j = 0
                for i in range(output_size):
                    if i == target:
                        continue
                    obj_val = solver_result[ j ][ 0 ]
                    if obj_val < smallestobj:
                        smallestobj = obj_val
                        bad_class = i
                        bad_exam = solver_result[ j ][ 1 ]
                    j += 1
                
                del var_list
                del model
                del counter

                elapsed_time = time.time() - t                                                                         
                print( 'Time:', elapsed_time, 'secs' )

                if bad_class == None:
                    return True

                return ( bad_exam, bad_class, smallestobj )
            else:
                bounds_are_computed = not len( self.nlb[ layerno + 1 ] ) == 0 and not len( self.nub[ layerno + 1 ] ) == 0
                if bounds_are_computed:
                    lbi = self.nlb[ layerno + 1 ]
                    ubi = self.nub[ layerno + 1 ]
                else:
                    lbi = np.array( [ -GRB.INFINITY ] * output_size )
                    ubi = np.array( [ GRB.INFINITY ] * output_size )
                
                model.setParam(GRB.Param.Threads, 1)
                model.update()

                t = time.time()
                if complete or not bounds_are_computed:
                    neurons = np.array( range( output_size ), np.int64 )
                else:
                    l = np.array( self.nlb[ layerno + 1 ] ) < 0
                    u = np.array( self.nub[ layerno+ 1 ] ) > 0
                    neurons = np.where( np.logical_and( l, u ) )[ 0 ]
                    neurons = np.int64( neurons )
                #neurons = np.array( [0], np.int64 ) 
                var_names = [ var_list[ var ].VarName for var in counter + neurons ]
                print( 'Recomputed Vars:', len( var_names ) )
                with multiprocessing.Pool(ncpus) as pool:
                    solver_result = pool.map( pool_func, var_names )
                del globals()[ 'global_model' ]
                for i in range( neurons.shape[ 0 ] ):
                    lbi[ neurons[ i ] ] = solver_result[ i ][ 0 ]
                    ubi[ neurons[ i ] ] = solver_result[ i ][ 1 ]
                elapsed_time = time.time() - t
                print( 'Time:', elapsed_time, 'secs' )
                if bounds_are_computed:
                    self.nub[ layerno + 1 ] = np.minimum( self.nub[ layerno + 1 ], ubi )
                    self.nlb[ layerno + 1 ] = np.maximum( self.nlb[ layerno + 1 ], lbi )
                else:
                    self.nub[ layerno + 1 ] = ubi
                    self.nlb[ layerno + 1 ] = lbi

            del var_list
            del model
            del counter

    def create_tf_sampling_net( self, net_file, is_trained_with_pytorch, ver_type ):
        from read_net_file import myConst, parseVec, extract_mean, extract_std, permutation, runRepl
        if ver_type == 'MILP':
            use_milp = True
        else:
            use_milp = False
        if ver_type == 'DeepPoly':
            use_deeppoly = True
        else:
            use_deeppoly = False
        
        mean = 0.0
        std = 0.0
        net = open(net_file,'r')
        x = tf.placeholder( tf.float64, [ None, self.input_size ], name='x_sampling' )
        num_images = tf.shape( x )[ 0 ]
        sampling_layers_size = tf.stack( ( num_images, -1 ), axis=0 )
        self.tf_sampling_x = x
        last_layer = None
        h,w,c = None, None, None
        is_conv = False

        tf_attack = tf.placeholder( x.dtype, shape=[None] )
        tf_attack_idx = self.input_size 
        relu_idx = 1 
        tf_nlb = []
        tf_nub = []
        tf_sampling_layers = []
       
        deeppoly_layer_args = []
        shapes = []
        while True:
            curr_line = net.readline()[:-1]
            if 'Normalize' in curr_line:
                mean = extract_mean(curr_line)
                std = extract_std(curr_line)
            elif curr_line in ["ReLU", "Affine"]:
                print(curr_line)
                W = None
                if (last_layer in ["Conv2D", "ParSumComplete", "ParSumReLU"]) and is_trained_with_pytorch:
                    W = myConst(permutation(parseVec(net), h, w, c).transpose())
                else:
                    W = myConst(parseVec(net).transpose())
                b = parseVec(net)
                if len( shapes ) == 0:
                    shapes.append( x.shape )
                #b = myConst(b.reshape([1, numel(b)]))
                b = myConst(b)
                if(curr_line=="Affine"):
                    x = tf.nn.bias_add(tf.matmul(tf.reshape(x, sampling_layers_size),W), b)
                    tf_sampling_layers.append( tf.reshape( x, shape=sampling_layers_size ) )
                    deeppoly_layer_args.append( ( 'Affine', W, b, None, None ) )
                    shapes.append( x.shape )
                elif(curr_line=="ReLU"):
                    x = tf.nn.bias_add(tf.matmul(tf.reshape(x, sampling_layers_size),W), b)
                    tf_sampling_layers.append( tf.reshape( x, shape=sampling_layers_size ) )
                    prev_layer_attack = tf_attack[ tf_attack_idx : tf_attack_idx + np.prod( x.shape[ 1 : ] ) ]
                    tf_attack_idx_st = tf_attack_idx + np.prod( x.shape[ 1 : ] )
                    tf_attack_idx = tf_attack_idx_en = tf_attack_idx_st + np.prod( x.shape[ 1 : ] )
                    tf_attack_layer = tf_attack[ tf_attack_idx_st : tf_attack_idx_en ]
                    x, x_bef_reshape, tf_lbi, tf_ubi = self.create_tf_layer( x, prev_layer_attack, tf_attack_layer, relu_idx, use_deeppoly )
                    tf_nlb.append( tf_lbi )
                    tf_nub.append( tf_ubi )
                    tf_sampling_layers.append( x_bef_reshape )
                    deeppoly_layer_args.append( ( 'Affine', W, b, tf_lbi, tf_ubi ) )
                    shapes.append( x.shape )
                    relu_idx += 1
                print("\tOutShape: ", x.shape)
                print("\tWShape: ", W.shape)
                print("\tBShape: ", b.shape)
            elif curr_line == "Conv2D":
                is_conv = True
                line = net.readline()
                args = None
                #print(line[-10:-3])
                start = 0
                if("ReLU" in line):
                    start = 5
                elif("Sigmoid" in line):
                    start = 8
                elif("Tanh" in line):
                    start = 5
                elif("Affine" in line):
                    start = 7
                if 'padding' in line:
                    args =  runRepl(line[start:-1], ["filters", "input_shape", "kernel_size", "stride", "padding"])
                else:
                    args = runRepl(line[start:-1], ["filters", "input_shape", "kernel_size"])

                W = myConst(parseVec(net))
                print("W shape", W.shape)
                #W = myConst(permutation(parseVec(net), h, w, c).transpose())
                b = None
                if("padding" in line):
                    if(args["padding"]==1):
                        padding_arg = "SAME"
                    else:
                        padding_arg = "VALID"
                else:
                    padding_arg = "VALID"

                if("stride" in line):
                    stride_arg = [1] + args["stride"] + [1]
                else:
                    stride_arg = [1,1,1,1]

                tf_out_shape = tf.stack( [ num_images ] + args["input_shape"], axis=0 )
                x = tf.reshape(x, tf_out_shape)
                if len( shapes ) == 0:
                    shapes.append( x.shape )

                x = tf.nn.conv2d(x, filter=W, strides=stride_arg, padding=padding_arg)
                b = myConst(parseVec(net))
                h, w, c = [ int( i ) for i in x.shape[ 1 : ] ]
                print("Conv2D", args, "W.shape:",W.shape, "b.shape:", b.shape)
                print("\tOutShape: ", x.shape)
                if("ReLU" in line):
                    x = tf.nn.bias_add(x, b)
                    tf_sampling_layers.append( tf.reshape( x, shape=sampling_layers_size ) )
                    prev_layer_attack = tf_attack[ tf_attack_idx : tf_attack_idx + np.prod( x.shape[ 1 : ] ) ]
                    tf_attack_idx_st = tf_attack_idx + np.prod( x.shape[ 1 : ] )
                    tf_attack_idx = tf_attack_idx_en = tf_attack_idx_st + np.prod( x.shape[ 1 : ] )
                    tf_attack_layer = tf_attack[ tf_attack_idx_st : tf_attack_idx_en ]
                    x, x_bef_reshape, tf_lbi, tf_ubi = self.create_tf_layer( x, prev_layer_attack, tf_attack_layer, relu_idx, use_deeppoly )
                    tf_sampling_layers.append( x_bef_reshape )
                    tf_nlb.append( tf_lbi )
                    tf_nub.append( tf_ubi )
                    deeppoly_layer_args.append( ( 'Conv2D', W, b, stride_arg, padding_arg, tf_lbi, tf_ubi ) )
                    shapes.append( x.shape )
                    relu_idx += 1
                elif("Affine" in line):
                    x = tf.nn.bias_add(x, b)
                    tf_sampling_layers.append( tf.reshape( x, shape=sampling_layers_size ) )
                    deeppoly_layer_args.append( ( 'Conv2D', W, b, stride_arg, padding_arg, None, None ) )
                    shapes.append( x.shape )
                else:
                    raise Exception("Unsupported activation: ", curr_line)
            elif curr_line == "":
                break
            else:
                raise Exception("Unsupported Operation: ", curr_line)
            last_layer = curr_line
        if 'ReLU' in last_layer:
            tf_nlb = tf_nlb[ : -1 ]
            tf_nub = tf_nub[ : -1 ]
        self.tf_nlb = tf_nlb
        self.tf_nub = tf_nub
        self.tf_attack = tf_attack
        self.tf_sampling_layers = tf_sampling_layers

        x_inf = tf.placeholder( tf.float64, shapes[ -1 ], name='x_backward_deeppoly_inf' )
        x_sup = tf.placeholder( tf.float64, shapes[ -1 ], name='x_backward_deeppoly_sup' )
        x_inf_cst = tf.placeholder( tf.float64, [ None, 1 ], name='x_backward_deeppoly_inf_cst' )
        x_sup_cst = tf.placeholder( tf.float64, [ None, 1 ], name='x_backward_deeppoly_sup_cst' )
        backsubstitute = [ ( x_inf, x_sup, x_inf_cst, x_sup_cst ) ]
        layer_types = [] 
        for i in range( len( deeppoly_layer_args )-1, -1, -1 ):
            if deeppoly_layer_args[i][0] == 'Affine':
                if i != 0:
                    nlb_nub = deeppoly_layer_args[ i - 1 ][ -2 : ]
                else:
                    nlb_nub = [ None, None ]
                out = self.create_deeppoly_backsubs_ffn( *deeppoly_layer_args[ i ][ 1 : -2 ], *nlb_nub, *backsubstitute[ 0 ] )
                out = list( out )
                out[ 0 ] = tf.reshape( out[ 0 ], [ -1 ] + shapes[ i ][ 1 : ].as_list() ) 
                out[ 1 ] = tf.reshape( out[ 1 ], [ -1 ] + shapes[ i ][ 1 : ].as_list() )
                backsubstitute.insert( 0, out )
            elif deeppoly_layer_args[i][0] == 'Conv2D':
                if i != 0:
                    nlb_nub = deeppoly_layer_args[ i - 1 ][ -2 : ]
                else:
                    nlb_nub = [ None, None ]
                out = self.create_deeppoly_backsubs_conv( shapes[ i ][ 1 : ].as_list(), *deeppoly_layer_args[ i ][ 1 : -2 ], *nlb_nub, *backsubstitute[ 0 ] )
                out = list( out )
                out[ 0 ] = tf.reshape( out[ 0 ], [ -1 ] + shapes[ i ][ 1 : ].as_list() ) 
                out[ 1 ] = tf.reshape( out[ 1 ], [ -1 ] + shapes[ i ][ 1 : ].as_list() ) 
                backsubstitute.insert( 0, out )
        self.backsubstitute_tens = backsubstitute

    def create_deeppoly_backsubs_conv( self, shape, W, b, stride_arg, padding_arg, tf_lbi, tf_ubi, inf, sup, inf_cst, sup_cst ):
        batch_size = tf.shape( inf )[ 0 ]
        deconv_shape = tf.stack( [ batch_size, *shape ] )

        deconv_inf = tf.nn.conv2d_transpose( inf, filter=W, output_shape=deconv_shape, strides=stride_arg, padding=padding_arg)
        deconv_sup = tf.nn.conv2d_transpose( sup, filter=W, output_shape=deconv_shape, strides=stride_arg, padding=padding_arg)

        mul = tf.tensordot( inf, b, axes=1 )
        reduce_dims =  list( range( 1, len ( mul.shape ) ) )
        deconv_inf_cst = inf_cst  + tf.reduce_sum( mul, reduce_dims )[ : , tf.newaxis ]
        mul = tf.tensordot( sup, b, axes=1 )
        deconv_sup_cst = sup_cst  + tf.reduce_sum( mul, reduce_dims )[ : , tf.newaxis ]

        if tf_lbi == None:
            return deconv_inf, deconv_sup, deconv_inf_cst, deconv_sup_cst
        else:
            deconv_inf_non_neg = tf.nn.relu( deconv_inf )
            deconv_inf_non_pos = -tf.nn.relu( -deconv_inf )

            deconv_sup_non_neg = tf.nn.relu( deconv_sup )
            deconv_sup_non_pos = -tf.nn.relu( -deconv_sup )

            relu_inf_non_neg, relu_inf_non_pos, _, relu_inf_cst = self.create_deeppoly_backsubs_relu( tf_lbi, tf_ubi, deconv_inf_non_neg, deconv_inf_non_pos, deconv_inf_cst, deconv_inf_cst )
            relu_inf = relu_inf_non_neg + relu_inf_non_pos
            relu_sup_non_pos, relu_sup_non_neg, _, relu_sup_cst = self.create_deeppoly_backsubs_relu( tf_lbi, tf_ubi, deconv_sup_non_pos, deconv_sup_non_neg, deconv_sup_cst, deconv_sup_cst )
            relu_sup = relu_sup_non_neg + relu_sup_non_pos
            return relu_inf, relu_sup, relu_inf_cst, relu_sup_cst

    def create_deeppoly_backsubs_ffn( self, W, b, tf_lbi, tf_ubi, inf, sup, inf_cst, sup_cst ):
        deconv_inf = tf.matmul( inf,  tf.transpose( W ) )
        deconv_sup = tf.matmul( sup, tf.transpose( W ) )

        mul = tf.tensordot( inf, b, axes=1 )
        reduce_dims =  list( range( 1, len ( mul.shape ) ) )
        deconv_inf_cst = inf_cst  + tf.reduce_sum( mul, reduce_dims )[ : , tf.newaxis ]
        mul = tf.tensordot( sup, b, axes=1 )
        deconv_sup_cst = sup_cst  + tf.reduce_sum( mul, reduce_dims )[ : , tf.newaxis ]

        if tf_lbi == None:
            return deconv_inf, deconv_sup, deconv_inf_cst, deconv_sup_cst
        else:
            deconv_inf_non_neg = tf.nn.relu( deconv_inf )
            deconv_inf_non_pos = -tf.nn.relu( -deconv_inf )

            deconv_sup_non_neg = tf.nn.relu( deconv_sup )
            deconv_sup_non_pos = -tf.nn.relu( -deconv_sup )

            relu_inf_non_neg, relu_inf_non_pos, _, relu_inf_cst = self.create_deeppoly_backsubs_relu( tf_lbi, tf_ubi, deconv_inf_non_neg, deconv_inf_non_pos, deconv_inf_cst, deconv_inf_cst )
            relu_inf = relu_inf_non_neg + relu_inf_non_pos
            relu_sup_non_pos, relu_sup_non_neg, _, relu_sup_cst = self.create_deeppoly_backsubs_relu( tf_lbi, tf_ubi, deconv_sup_non_pos, deconv_sup_non_neg, deconv_sup_cst, deconv_sup_cst )
            relu_sup = relu_sup_non_neg + relu_sup_non_pos
            return relu_inf, relu_sup, relu_inf_cst, relu_sup_cst

    def create_deeppoly_backsubs_relu( self, tf_lbi, tf_ubi, inf, sup, inf_cst, sup_cst ):
        affine_shape = tf.shape( inf )
        num_images = affine_shape[ 0 ]
        sampling_layers_size = tf.stack( ( num_images, -1 ), axis=0 )

        prev_layer_inf = tf.reshape( inf, shape=sampling_layers_size )
        prev_layer_sup = tf.reshape( sup, shape=sampling_layers_size )
        numel = tf.shape( prev_layer_inf )[ 1 ]

        # nub < 0 => y = 0
        tf_out_zeros_count = tf.math.count_nonzero( tf_ubi <= 0, dtype=tf.int32 )
        tf_out_zeros_shape = tf.stack( ( num_images, tf_out_zeros_count ), axis=0 )
        tf_out_zeros = tf.zeros( shape=tf_out_zeros_shape, dtype=inf.dtype ) 

        # nlb > 0 => y = x
        tf_out_x_inf = tf.gather( prev_layer_inf, tf.where( tf_lbi >= 0 )[ :, 0 ], axis=1 )
        tf_out_x_sup = tf.gather( prev_layer_sup, tf.where( tf_lbi >= 0 )[ :, 0 ], axis=1 )

        # remaining_idx calculations
        remaining_idx_full = tf.logical_and( tf_lbi < 0, tf_ubi > 0 )
        remaining_idx = tf.where( remaining_idx_full )[ : , 0 ]
        tf_lbi_remaining = tf.gather( tf_lbi, remaining_idx )
        tf_ubi_remaining = tf.gather( tf_ubi, remaining_idx )
        prev_layer_inf = tf.gather( prev_layer_inf, remaining_idx, axis=1 )
        prev_layer_sup = tf.gather( prev_layer_sup, remaining_idx, axis=1 )

        slope = tf_ubi_remaining / ( tf_ubi_remaining - tf_lbi_remaining )
        intercept = -slope * tf_lbi_remaining
        
        reduce_dims = list( range( 1, len( prev_layer_sup.shape ) ) )
        sup_cst += tf.reduce_sum( prev_layer_sup * intercept[ tf.newaxis, : ], axis=reduce_dims )[ : , tf.newaxis ]

        # area decision:
        tf_ubi_abs = tf.abs( tf_ubi )
        tf_lbi_abs = tf.abs( tf_lbi )

        b1_eliminated_idx_full = tf_ubi_abs < tf_lbi_abs
        b1_eliminated_idx = tf.gather( b1_eliminated_idx_full, remaining_idx )
        b1_eliminated_idx = tf.where( b1_eliminated_idx )[ : , 0 ]

        b1_eliminated_output_inf = tf.zeros( shape=( tf.stack( ( num_images, numel ), axis=0 ) ), dtype=inf.dtype )
        b1_eliminated_output_inf = tf.gather( b1_eliminated_output_inf, b1_eliminated_idx, axis=1 )

        b3_eliminated_idx_full = tf_ubi_abs >= tf_lbi_abs
        b3_eliminated_idx = tf.gather( b3_eliminated_idx_full, remaining_idx )
        b3_eliminated_idx = tf.where( b3_eliminated_idx )[ : , 0 ]

        b3_eliminated_output_inf = prev_layer_inf
        b3_eliminated_output_inf = tf.gather( b3_eliminated_output_inf, b3_eliminated_idx, axis=1 )

        b2_output_sup = prev_layer_sup * slope 

        out_combined_inf = tf.concat( ( tf_out_zeros, tf_out_x_inf, b3_eliminated_output_inf, b1_eliminated_output_inf ), axis=1 )
        zero_idx = tf.where( tf_ubi <= 0 ) [ : , 0 ]
        x_idx = tf.where( tf_lbi >= 0 ) [ : , 0 ]
        b3_elim_idx = tf.where( tf.logical_and( remaining_idx_full, b3_eliminated_idx_full ) ) [ : , 0 ]
        b1_elim_idx = tf.where( tf.logical_and( remaining_idx_full, b1_eliminated_idx_full ) ) [ : , 0 ]
        idx = tf.argsort( tf.concat( ( zero_idx, x_idx, b3_elim_idx, b1_elim_idx ), axis=0 ) )
        out_combined_reordered_inf = tf.gather( out_combined_inf, idx, axis=1 )
        out_combined_reordered_reshaped_inf = tf.reshape( out_combined_reordered_inf, shape=( affine_shape ) )

        out_combined_sup = tf.concat( ( tf_out_zeros, tf_out_x_sup, b2_output_sup ), axis=1 )
        idx = tf.argsort( tf.concat( ( zero_idx, x_idx, remaining_idx ), axis=0 ) )
        out_combined_reordered_sup = tf.gather( out_combined_sup, idx, axis=1 )
        out_combined_reordered_reshaped_sup = tf.reshape( out_combined_reordered_sup, shape=( affine_shape ) )

        return out_combined_reordered_reshaped_inf, out_combined_reordered_reshaped_sup, inf_cst, sup_cst

    def create_tf_layer( self, affine, prev_layer_attack, tf_attack_layer, relu_idx, use_deeppoly ):
        tf_lbi = tf.placeholder( affine.dtype, shape=[None], name='lbi_%i' % relu_idx )
        tf_ubi = tf.placeholder( affine.dtype, shape=[None], name='ubi_%i' % relu_idx )

        affine_shape = tf.shape( affine )
        num_images = affine_shape[ 0 ]
        sampling_layers_size = tf.stack( ( num_images, -1 ), axis=0 )

        prev_layer = tf.reshape( affine, shape=sampling_layers_size )
        # nub < 0 => y = 0
        tf_out_zeros_count = tf.math.count_nonzero( tf_ubi <= 0, dtype=tf.int32 )
        tf_out_zeros_shape = tf.stack( ( num_images, tf_out_zeros_count ), axis=0 )
        tf_out_zeros = tf.zeros( shape=tf_out_zeros_shape, dtype=affine.dtype )

        # nlb > 0 => y = x
        tf_out_x = tf.gather( prev_layer, tf.where( tf_lbi >= 0 )[ :, 0 ], axis=1 )

        # remaining_idx calculations
        remaining_idx_full = tf.logical_and( tf_lbi < 0, tf_ubi > 0 )
        remaining_idx = tf.where( remaining_idx_full )[ : , 0 ]
        tf_lbi_remaining = tf.gather( tf_lbi, remaining_idx )
        tf_ubi_remaining = tf.gather( tf_ubi, remaining_idx )
        prev_layer_attack = tf.gather( prev_layer_attack, remaining_idx )
        tf_attack_layer = tf.gather( tf_attack_layer, remaining_idx )
        prev_layer = tf.gather( prev_layer, remaining_idx, axis=1 )

        # remove y >= 0
        slope = tf_ubi_remaining / ( tf_ubi_remaining - tf_lbi_remaining )
        intercept = -slope * tf_lbi_remaining
        b2 = prev_layer_attack * slope + intercept
        b1 = prev_layer_attack
        b3_eliminated_output = tf.clip_by_value( tf_attack_layer, b1, b2 )

        region = b2 - b1
        low_dist = b3_eliminated_output - b1
        region = tf.clip_by_value( region, 1e-6, region )
        low_dist = tf.clip_by_value( low_dist, 5e-7, low_dist )
        high_dist = region - low_dist
        low_dist /= region
        high_dist /= region

        b2_output = prev_layer * slope + intercept
        b1_output = prev_layer
        b3_eliminated_output = low_dist * b2_output +  high_dist * b1_output

        # remove y >= x
        b1 = 0
        b1_eliminated_output = tf.clip_by_value( tf_attack_layer, b1, b2 )

        region = b2 - b1
        low_dist = b1_eliminated_output - b1
        region = tf.clip_by_value( region, 1e-6, region )
        low_dist = tf.clip_by_value( low_dist, 5e-7, low_dist )
        high_dist = region - low_dist
        low_dist /= region
        high_dist /= region

        b1_eliminated_output = low_dist * b2_output

        tf_ubi_abs = tf.abs( tf_ubi )
        tf_lbi_abs = tf.abs( tf_lbi )

        if use_deeppoly:
            # abs( ubi ) > abs( lbi ) => remove y >= 0 
            b3_eliminated_idx_full = tf_ubi_abs >= tf_lbi_abs
            b3_eliminated_idx = tf.gather( b3_eliminated_idx_full, remaining_idx )
        else:
            # b3 < b1 => remove y >= 0 
            b3_eliminated_idx = b1_output >= 0

        if use_deeppoly:
            # abs( ubi ) < abs( lbi ) => remove y >= x 
            b1_eliminated_idx_full = tf_ubi_abs < tf_lbi_abs
            b1_eliminated_idx = tf.gather( b1_eliminated_idx_full, remaining_idx )
        else:
            # b3 > b1 => remove y >= x
            b1_eliminated_idx = b1_output < 0

        remaining_output = tf.where( b1_eliminated_idx, b1_eliminated_output, b3_eliminated_output )

        out_combined = tf.concat( ( tf_out_zeros, tf_out_x, remaining_output ), axis=1 )
        zero_idx = tf.where( tf_ubi <= 0 ) [ : , 0 ]
        x_idx = tf.where( tf_lbi >= 0 ) [ : , 0 ]
        idx = tf.argsort( tf.concat( ( zero_idx, x_idx, remaining_idx ), axis=0 ) )
        out_combined_reordered = tf.gather( out_combined, idx, axis=1 )
        out_combined_reordered_reshaped = tf.reshape( out_combined_reordered, shape=( affine_shape ) )

        return out_combined_reordered_reshaped, out_combined_reordered, tf_lbi, tf_ubi

    def extract_deeppoly_backsub( self, target, batch_size=50 ):
        feed_dict = {}
        ncpus = os.sysconf("SC_NPROCESSORS_ONLN")
        
        for i in range( len( self.tf_nlb ) + 1 ):
            is_final_layer = ( i == len( self.tf_nlb ) )
            s = time.time()
            if i != 0:
                feed_dict[ self.tf_nlb[ i - 1 ] ] = self.nlb[ i ]
                feed_dict[ self.tf_nub[ i - 1 ] ] = self.nub[ i ]

            size = self.backsubstitute_tens[ i + 1 ][ 0 ].shape.as_list()[ 1 : ]
            layer_size_full = np.prod( size )
            if is_final_layer:
                recompute_idx = [ j for j in range( layer_size_full ) if j != target ]
            else:
                recompute_idx = np.where( np.logical_and( self.nlb[ i + 1 ] < 0, self.nub[ i + 1 ] > 0 ) )[ 0 ]
            layer_size = len( recompute_idx )
            if layer_size == 0:
                continue
            
            feed_dict[ self.backsubstitute_tens[ i + 1 ][ 2 ] ] = np.zeros( ( batch_size, 1 ) )
            feed_dict[ self.backsubstitute_tens[ i + 1 ][ 3 ] ] = np.zeros( ( batch_size, 1 ) )

            lb_layer = []
            ub_layer = []
            out = [ np.zeros( ( 0, self.input_size ) ),  np.zeros( ( 0, self.input_size ) ), np.zeros( ( 0, 1 ) ), np.zeros( ( 0, 1 ) ) ]
            j = -1
            for j in range( int( layer_size / batch_size ) ):
                eye_input = np.zeros( ( batch_size, layer_size_full ) )
                idx = recompute_idx[ j * batch_size : ( j + 1 ) * batch_size ]
                idx = [ tuple( np.arange(batch_size) ), tuple( idx ) ]
                eye_input[ idx ] = 1
                if is_final_layer:
                    eye_input[ : , target ] = -1
                feed_dict[ self.backsubstitute_tens[ i + 1 ][ 0 ] ] = eye_input.reshape( [-1] + size )
                feed_dict[ self.backsubstitute_tens[ i + 1 ][ 1 ] ] = eye_input.reshape( [-1] + size )
                out_batch = self.sess.run( self.backsubstitute_tens[ 0 ], feed_dict=feed_dict )
                out_batch[ 0 ] = out_batch[ 0 ].reshape( out_batch[ 0 ].shape[ 0 ], -1 ) 
                out_batch[ 1 ] = out_batch[ 1 ].reshape( out_batch[ 1 ].shape[ 0 ], -1 )
                for k in range( 4 ):
                    out[ k ] = np.concatenate( ( out[ k ], out_batch[ k ] ), axis=0 )
            j += 1
            if layer_size - j * batch_size > 0 :
                eye_input = np.zeros( ( layer_size - j * batch_size, layer_size_full ) )
                idx = recompute_idx[ j * batch_size : layer_size ]
                idx = [ tuple( np.arange( eye_input.shape[ 0 ] ) ), tuple( idx ) ]
                eye_input[ idx ] = 1
                if is_final_layer:
                    eye_input[ : , target ] = -1
                eye_input = eye_input.reshape( [-1] + size )

                feed_dict[ self.backsubstitute_tens[ i + 1 ][ 0 ] ] = eye_input
                feed_dict[ self.backsubstitute_tens[ i + 1 ][ 1 ] ] = eye_input
                feed_dict[ self.backsubstitute_tens[ i + 1 ][ 2 ] ] = np.zeros( ( eye_input.shape[ 0 ], 1 ) )
                feed_dict[ self.backsubstitute_tens[ i + 1 ][ 3 ] ] = np.zeros( ( eye_input.shape[ 0 ], 1 ) )
                out_batch = self.sess.run( self.backsubstitute_tens[ 0 ], feed_dict=feed_dict )
                out_batch[ 0 ] = out_batch[ 0 ].reshape( out_batch[ 0 ].shape[ 0 ], -1 ) 
                out_batch[ 1 ] = out_batch[ 1 ].reshape( out_batch[ 1 ].shape[ 0 ], -1 )
                for k in range( 4 ):
                    out[ k ] = np.concatenate( ( out[ k ], out_batch[ k ] ), axis=0 )

            global global_model, global_eq, global_xs, global_get_example
            global_model = self.model
            global_eq = out
            global_get_example = is_final_layer
            with multiprocessing.Pool(ncpus) as pool:
                solver_result = pool.map( pool_func_deeppoly, list( range ( layer_size ) ) )
            del globals()[ 'global_model' ]
            del globals()[ 'global_eq' ]
            del globals()[ 'global_get_example' ]

            if is_final_layer:
                lbi, ubi, bad_exams = zip( *solver_result )
            else:
                lbi, ubi = zip( *solver_result )
            lbi = np.array( lbi )
            ubi = np.array( ubi )
            
            if is_final_layer:
                bad_class = np.argmax( ubi )
                smallestobj = -ubi[ bad_class ]
                bad_exam = bad_exams[ bad_class ]
                bad_exam_eq = -out[ 1 ][ bad_class ] 
                bad_exam_eq_cst = -out[ 3 ][ bad_class ]
                bad_exam_eq = ( bad_exam_eq, bad_exam_eq_cst )
                if bad_class >= target:
                    bad_class += 1
                if smallestobj >= 0:
                    return True
                return ( bad_exam_eq, bad_exam, bad_class, smallestobj )
            else:
                self.nub[ i + 1 ][ recompute_idx ] = np.minimum( self.nub[ i + 1 ][ recompute_idx ], ubi )
                self.nlb[ i + 1 ][ recompute_idx ] = np.maximum( self.nlb[ i + 1 ][ recompute_idx ], lbi )

            for j in range( 4 ):
                del feed_dict[ self.backsubstitute_tens[ i + 1 ][ j ] ]
            print( 'Layer', i,':', layer_size,'/', layer_size_full, time.time() - s ,'secs' )
    
    def calc_tf_sampling_net( self, example, sample_batch ): 
        feed_dict = { self.tf_sampling_x: sample_batch, self.tf_attack: example }
        for i in range( len( self.tf_nlb ) ):
            feed_dict[ self.tf_nlb[ i ] ] = self.nlb[ i + 1 ]
            feed_dict[ self.tf_nub[ i ] ] = self.nub[ i + 1 ]
        out = self.sess.run( self.tf_sampling_layers, feed_dict=feed_dict )
        attack = np.concatenate( [ sample_batch ] + out, axis=1 )
        is_attack = np.logical_not( np.argmax( out[ -1 ], axis=1 ) == self.y_tar )
        return is_attack, attack

    def tf_lp_sampling( self, attack, samples, vertype, batch_size=50 ):
        ts = []
        for i in range( int( samples.shape[ 0 ] / batch_size ) ):
            t = samples[ i * batch_size : ( i + 1 ) * batch_size ]
            which,_ = self.calc_tf_sampling_net( attack, t )
            ts.append( t[ which ] )
        i = i + 1
        t = samples[ i * batch_size : ]
        which,_ = self.calc_tf_sampling_net( attack, t )
        ts.append( t[ which ] )
        ts = np.concatenate( ts, axis=0 )
        return ts

    def shrink_poly( self, nn, ver_type, target ):
        min_bound = 1.0 / ( 2**10 )
        eps = min_bound
        sol = None
        print( 'Initial search' )
        while eps < 1:
            bound = self.shrink_poly_by_eps( nn, ver_type, target, eps )
            if bound == 0:
                break
            print( 'Eps', eps, 'Bound', bound )
            eps *= 2.0
        if eps == None:
            eps = 1.0
        sol = eps
        start_eps = eps / 4.0
        print( 'Initial eps', sol )
        while start_eps >= min_bound:
            print( 'Start eps', start_eps, 'Sol', sol )
            while start_eps >= min_bound:
                bound = self.shrink_poly_by_eps( nn, ver_type, target, eps - start_eps )
                if bound == 0:
                    sol = eps - start_eps
                    break 
                else:
                    print( 'Eps', eps - start_eps, 'Bound', bound )
                start_eps /= 2.0 
            eps -= start_eps
            start_eps /= 2.0

        print( 'Sol:', sol ) 
        
        self.model = self.create_shrink_model( sol )
        self.xs = [ self.model.getVarByName( 'x%i' % i ) for i in range( self.input_size ) ]
        self.invalidate_calc()

        return sol 
            
    def create_shrink_model( self, eps ):
        center = np.median( self.data, axis=0 )
        model = Model( 'InitPoint' )
        model.setParam( 'OutputFlag', 0 )

        for var in self.model.getVars():
            if var.LB is -GRB.INFINITY or var.UB is GRB.INFINITY:
                assert False
            var_id = int( var.VarName[ 1 : ] )
            eps_lw = center[ var_id ] - var.LB 
            eps_up = var.UB - center[ var_id ]

            var_copy = model.addVar(vtype=GRB.CONTINUOUS, lb=var.LB + eps * eps_lw, ub=var.UB - eps * eps_up, name=var.VarName)
        model.update()

        for constr in self.model.getConstrs():
            constr_new = LinExpr()
            coefs = []
            dist = 0
            for var in self.model.getVars():
                var_new = model.getVarByName( var.VarName )
                var_id = int( var.VarName[ 1 : ] )
                coef = self.model.getCoeff( constr, var )
                dist += coef * center[ var_id ] 
                constr_new += coef * var_new
                coefs.append( coef )
            dist -= constr.RHS
            dist = np.abs( dist )
            norm = np.linalg.norm( coefs )
            coefs = np.array( coefs ) / norm
            dist /= norm
            
            i = 0
            for var in self.model.getVars():
                var_new = model.getVarByName( var.VarName )
                constr_new += coefs[ i ] * var_new
                i += 1
            if constr.Sense == '<':
                constr_new += eps * dist
                model.addConstr(constr_new, GRB.LESS_EQUAL, constr.RHS/norm, constr.ConstrName )
            elif constr.Sense == '>':
                constr_new -= eps * dist
                model.addConstr(constr_new, GRB.GREATER_EQUAL, constr.RHS/norm, constr.ConstrName )
            else:
                assert False
        
        model.update()
        return model

    def shrink_poly_by_eps( self, nn, ver_type, target, eps ):
        model = self.create_shrink_model( eps )
        
        old_model = self.model
        old_xs = self.xs
        old_obox = self.obox
        old_ubox = self.ubox
        old_nlb = [ np.array( bound ).copy() for bound in self.nlb ]
        old_nub = [ np.array( bound ).copy() for bound in self.nub ]

        self.obox = None
        self.ubox = None
        self.nlb[ 0 ] = []
        self.nub[ 0 ] = []
        self.model = model
        self.xs = [ model.getVarByName( 'x%i' % i ) for i in range( self.input_size ) ]

        olb, oub = self.overapprox_box()
        output = self.lp_verification( nn, ver_type, target )
        
        self.obox = old_obox
        self.ubox = old_ubox
        self.nlb = old_nlb
        self.nub = old_nub
        self.model = old_model
        self.xs = old_xs
        
        del model
        
        if output == True:
            return 0
        else:
            return output[-1]
    
    def fit_plane( self, pos_ex, neg_ex ):

        # Create Dataset
        X = np.concatenate( ( pos_ex, neg_ex ) )
        zeros = np.zeros( neg_ex.shape[ 0 ] )
        ones = np.ones( pos_ex.shape[ 0 ] )
        y = np.concatenate( ( ones, zeros ) )

        # Shuffle it
        order = np.arange( X.shape[ 0 ] )
        np.random.shuffle( order )
        X = X[ order, : ]
        y = y[ order ]

        # SVM
        db_size = y.shape[ 0 ]
        ones = np.sum( y )
        zeros = db_size - ones

        #zeros_C = 2 * zeros / db_size
        #ones_C = ones / db_size
        #class_weight = { 0: zeros_C, 1: ones_C }
        #clf = LinearSVC( tol=5e-5, class_weight=class_weight, C=10, max_iter=2000 )
        clf = LogisticRegression( class_weight='balanced', max_iter=2000 )
        sample_weights = np.zeros( y.shape )
        sample_weights[ y == 0 ] = 100
        sample_weights[ y == 1 ] = 1
        clf.fit( X, y, sample_weights )

        W = clf.coef_
        b = clf.intercept_
        y_pred = clf.predict( X )

        bad_classification = np.where( np.logical_not( y == y_pred ) ) [ 0 ]
        print( 'Missclassifications: ', bad_classification.shape[ 0 ], '/', db_size )
        
        del clf

        return ( W, b )

    def __del__( self ):
        del self.model
        del self.xs
class LP_verifier:
    def __init__( self, attack, gurobi_model, gurobi_xs, nn, nlb, nub, ver_type ):
        self.gurobi_model = gurobi_model
        self.attack = attack

        if ver_type == 'MILP':
            use_milp = True
        else:
            use_milp = False
        if ver_type == 'DeepPoly':
            use_deeppoly = True
        else:
            use_deeppoly = False

        input_size = len( gurobi_xs )
        self.input_size = input_size
        gurobi_model.update()

        self.W = np.eye( input_size + 1 )
        for layer_counter in range( 2 * nn.numlayer - 1 ):
            t = time.time()
            layerno = int( layer_counter / 2 )
            relu_needed = [ 0 ] * ( layerno + 1 )
            for i in range( layerno ):
                relu_needed[ i ] = 1
            if layer_counter % 2 == 1:
                relu_needed[ -1 ] = 1
            krelu_groups = [ None ]*(layerno+1)
            if use_deeppoly:
                deeppoly = [1]*(layerno+1)
            else:
                deeppoly = False
            print( 'Layer:', layerno+1, 'MILP:', use_milp, 'DeepPoly:', deeppoly )
            counter, var_list, model = create_model(nn, nlb[ 0 ], nub[ 0 ], nlb[ 1 : ], nub[ 1 : ], krelu_groups, layerno + 1, use_milp, relu_needed, deeppoly)
            num_vars = len( var_list )
            output_size = num_vars - counter

            for constr in gurobi_model.getConstrs():
                constr_new = LinExpr()
                for i in range( input_size ):
                    coef = gurobi_model.getCoeff( constr, gurobi_xs[ i ] )
                    constr_new += coef * var_list[ i ]
                model.addConstr( constr_new, constr.Sense, constr.RHS )
            model.update()

            var_dict = {}
            for i in range( num_vars ):
                name = var_list[ i ].VarName
                var_dict[ name ] = i
            
            W = np.zeros( ( counter + 1, num_vars + 1 ) )
            if layer_counter % 2 == 0:
                # Handle EQ constraints
                eq_constrs = {}
                for constr in model.getConstrs():
                    if not constr.Sense == '=':
                        continue
                    for i in range( output_size ):
                        coef = model.getCoeff( constr, var_list[ counter + i ] )
                        if not coef == 0.0:
                            assert ( not i in eq_constrs )
                            eq_constrs[ i ] = constr

                for var_id, constr in eq_constrs.items():
                    linexpr = model.getRow( constr )
                    var_name = var_list[ counter + var_id ].VarName
                    var_coef = None
                    
                    coefs = np.zeros( counter + 1 )
                    for i in range( linexpr.size() ):
                        name = linexpr.getVar( i ).VarName
                        coef = linexpr.getCoeff( i )
                        if name == var_name:
                            var_coef = coef
                            continue
                        index = var_dict[ name ]
                        coefs[ index ] = coef
                    coefs[ -1 ] = linexpr.getConstant( ) - constr.RHS
                    coefs /= -var_coef
                    W[ : , counter + var_id ] = coefs 

            else:
                ubi = np.array( nub[ layerno + 1 ] )
                lbi = np.array( nlb[ layerno + 1 ] )

                lp_output = attack[ counter : num_vars ]
                
                # ub < 0
                ubi_neg_idx = ubi <= 0 # W = 0, as output = 0
                remaining_idx = ubi > 0
                
                # lb > 0 
                lbi_pos_idx = np.logical_and( lbi >= 0, remaining_idx )
                lbi_pos_idx_out = counter + np.where( lbi_pos_idx )[ 0 ]
                lbi_pos_idx_in = lbi_pos_idx_out - output_size
                W[ lbi_pos_idx_in, lbi_pos_idx_out ] = 1 # y = x
                remaining_idx = np.logical_and( remaining_idx, lbi < 0 )
                    
                if use_deeppoly:
                    # abs( ubi ) > abs( lbi ) => remove y >= 0 
                    b3_eliminated_idx = np.logical_and( np.abs( ubi ) >= np.abs( lbi ), remaining_idx )
                else:
                    # b3 < b1 => remove y >= 0 
                    b1_full = attack[ counter - output_size : counter ]
                    b3_full = np.zeros( lp_output.shape )
                    b3_eliminated_idx = np.logical_and( b3_full <= b1_full, remaining_idx )

                # remove y >= 0
                b3_eliminated_output = lp_output[ b3_eliminated_idx ]
                slope = ubi[ b3_eliminated_idx ] / ( ubi[ b3_eliminated_idx ] - lbi[ b3_eliminated_idx ] )
                intercept = -slope * lbi[ b3_eliminated_idx ]
                prev_layer = attack[ counter - output_size : counter ]
                b2 = prev_layer[ b3_eliminated_idx ] * slope + intercept
                b1 = prev_layer[ b3_eliminated_idx ]
                b3_eliminated_output = np.maximum( b1, b3_eliminated_output )
                b3_eliminated_output = np.minimum( b2, b3_eliminated_output )
                region = b2 - b1
                low_dist = b3_eliminated_output - b1
                high_dist = b2 - b3_eliminated_output
                low_dist /= region
                high_dist /= region
                low_dist[ np.isnan( low_dist ) ] = 1
                high_dist[ np.isnan( high_dist ) ] = 0
                b3_eliminated_idx_out = counter + np.where( b3_eliminated_idx )[ 0 ]
                b3_eliminated_idx_in = b3_eliminated_idx_out - output_size
                W[ b3_eliminated_idx_in, b3_eliminated_idx_out ] = low_dist * slope
                W[ -1, b3_eliminated_idx_out ] = low_dist * intercept
                W[ b3_eliminated_idx_in, b3_eliminated_idx_out ] += high_dist
                if use_deeppoly:
                    remaining_idx = np.logical_and( remaining_idx, np.abs( ubi ) < np.abs( lbi ) )
                else:
                    remaining_idx = np.logical_and( remaining_idx, b3_full > b1_full )

                # abs( ubi ) < abs( lbi ) and use_deeppoly or b3 > b1 => remove y >= x 
                b1_eliminated_idx = remaining_idx
                b1_eliminated_output = lp_output[ b1_eliminated_idx ]
                slope = ubi[ b1_eliminated_idx ] / ( ubi[ b1_eliminated_idx ] - lbi[ b1_eliminated_idx ] )
                intercept = -slope * lbi[ b1_eliminated_idx ]
                prev_layer = attack[ counter - output_size : counter ]
                b2 = prev_layer[ b1_eliminated_idx ] * slope + intercept
                b1 = np.zeros( slope.shape[ 0 ] )
                b1_eliminated_output = np.maximum( b1, b1_eliminated_output )
                b1_eliminated_output = np.minimum( b2, b1_eliminated_output )
                region = b2 - b1
                low_dist = b1_eliminated_output - b1
                high_dist = b2 - b1_eliminated_output
                low_dist /= region
                high_dist /= region
                low_dist[ np.isnan( low_dist ) ] = 1
                high_dist[ np.isnan( high_dist ) ] = 0
                b1_eliminated_idx_out = counter + np.where( b1_eliminated_idx )[ 0 ]
                b1_eliminated_idx_in = b1_eliminated_idx_out - output_size
                W[ b1_eliminated_idx_in, b1_eliminated_idx_out ] = low_dist * slope
                W[ -1, b1_eliminated_idx_out ] = low_dist * intercept

            W[ : counter, : counter ] = np.eye( counter )
            W[ -1, -1 ] = 1
            self.W = np.matmul( self.W, W )
            
            test = np.concatenate( ( attack[ : input_size], np.ones((1,)) ) )
            test = np.matmul( test, self.W )
            test = test[ : -1 ]
            dist = np.abs( test - attack[ : test.shape[ 0 ] ] )
            #if not np.all( dist < 1e-4 ):
            #    where = np.where( dist < 1e-4 )
            #    import pdb; pdb.set_trace()
            print( np.max( dist[counter:] ) )
            print( np.average( dist[counter:] ) )
            
            elapsed_time = time.time() - t                                                                         
            print( 'Time:', elapsed_time, 'secs' )

            if layerno == nn.numlayer - 1:
                self.output_start = counter
                
                '''
                # Check correctness
                self.model = model
                self.var_list = var_list
                return
                '''

            del var_list
            del model

    def check_samples( self, samples, target ):
        samples_ex = np.concatenate( ( samples, np.ones( ( samples.shape[ 0 ], 1 ) ) ), axis=1 )
        outputs = np.matmul( samples_ex, self.W[ : , self.output_start : -1 ] )
        outputs = np.argmax( outputs, axis=1 )
        verified_samples = samples[ np.logical_not( target == outputs ), : ]
        return verified_samples
    
    def get_x0( self, x0, target, attack_class ):
        model = self.gurobi_model.copy()
        output_size = self.W.shape[ 1 ] - self.output_start - 1
        vars = [ target, attack_class ]

        for i in vars:
            o = model.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name='o%i' % i)
            col = self.W[ : , self.output_start + i ]
            constr = LinExpr()
            constr += -1.0 * o
            for j in range( col.shape[ 0 ] - 1 ):
                constr += col[ j ] * model.getVarByName( 'x%i' % j )
            model.addConstr( constr, GRB.EQUAL, -col[ -1 ] )
        model.update()

        constr = LinExpr()
        constr += model.getVarByName( 'o%i' % attack_class ) - model.getVarByName( 'o%i' % target )
        model.addConstr( constr, GRB.GREATER_EQUAL, 0 )

        obj = LinExpr()
        for i in range( self.input_size ):
            if self.attack[ i ] - x0[ i ] > 0:
                obj += -model.getVarByName( 'x%i' % i )
            else:
                obj += model.getVarByName( 'x%i' % i )
        model.setObjective(obj,GRB.MAXIMIZE)
        model.optimize()
        if model.SolCount==0:
            assert False
        x0_out = np.zeros( self.input_size )
        for i in range( self.input_size ):
            x0_out[ i ] = model.getVarByName( 'x%i' % i ).x
        return x0_out
