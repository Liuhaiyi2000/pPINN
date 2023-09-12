
import tensorflow as tf
import numpy as np
import scipy.io
from pyDOE import lhs
import time
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import scipy.io as scio

class PhysicsInformedNN:
    # Initialize the class
    def __init__(self, X0, u0, X_f, X_b1, u1, X_b2, u2, X_b3, u3, X_b4, u4, layers, lb, ub):
        
        self.lb = lb
        self.ub = ub
    
        self.x0 = X0[:, 0:1]
        self.y0 = X0[:, 1:2]
        self.t0 = X0[:, 2:3]
        self.q0 = X0[:, 3:4]

        self.xb1 = X_b1[:, 0:1]
        self.yb1 = X_b1[:, 1:2]
        self.tb1 = X_b1[:, 2:3]
        self.qb1 = X_b1[:, 3:4]

        self.xb2 = X_b2[:, 0:1]
        self.yb2 = X_b2[:, 1:2]
        self.tb2 = X_b2[:, 2:3]
        self.qb2 = X_b2[:, 3:4]

        self.xb3 = X_b3[:, 0:1]
        self.yb3 = X_b3[:, 1:2]
        self.tb3 = X_b3[:, 2:3]
        self.qb3 = X_b3[:, 3:4]

        self.xb4 = X_b4[:, 0:1]
        self.yb4 = X_b4[:, 1:2]
        self.tb4 = X_b4[:, 2:3]
        self.qb4 = X_b4[:, 3:4]

        self.u0 = u0
        self.u1 = u1
        self.u2 = u2
        self.u3 = u3
        self.u4 = u4

        self.x_f = X_f[:, 0:1]
        self.y_f = X_f[:, 1:2]
        self.t_f = X_f[:, 2:3]
        self.q_f = X_f[:, 3:4]

        self.layers = layers
        self.weights, self.biases = self.initialize_NN(layers)

        self.x0_tf = tf.placeholder(tf.float32, shape=[None, self.x0.shape[1]])
        self.y0_tf = tf.placeholder(tf.float32, shape=[None, self.y0.shape[1]])
        self.t0_tf = tf.placeholder(tf.float32, shape=[None, self.t0.shape[1]])
        self.q0_tf = tf.placeholder(tf.float32, shape=[None, self.q0.shape[1]])

        self.xb1_tf = tf.placeholder(tf.float32, shape=[None, self.xb1.shape[1]])
        self.yb1_tf = tf.placeholder(tf.float32, shape=[None, self.yb1.shape[1]])
        self.tb1_tf = tf.placeholder(tf.float32, shape=[None, self.tb1.shape[1]])
        self.qb1_tf = tf.placeholder(tf.float32, shape=[None, self.qb1.shape[1]])

        self.xb2_tf = tf.placeholder(tf.float32, shape=[None, self.xb2.shape[1]])
        self.yb2_tf = tf.placeholder(tf.float32, shape=[None, self.yb2.shape[1]])
        self.tb2_tf = tf.placeholder(tf.float32, shape=[None, self.tb2.shape[1]])
        self.qb2_tf = tf.placeholder(tf.float32, shape=[None, self.qb2.shape[1]])

        self.xb3_tf = tf.placeholder(tf.float32, shape=[None, self.xb3.shape[1]])
        self.yb3_tf = tf.placeholder(tf.float32, shape=[None, self.yb3.shape[1]])
        self.tb3_tf = tf.placeholder(tf.float32, shape=[None, self.tb3.shape[1]])
        self.qb3_tf = tf.placeholder(tf.float32, shape=[None, self.qb3.shape[1]])

        self.xb4_tf = tf.placeholder(tf.float32, shape=[None, self.xb4.shape[1]])
        self.yb4_tf = tf.placeholder(tf.float32, shape=[None, self.yb4.shape[1]])
        self.tb4_tf = tf.placeholder(tf.float32, shape=[None, self.tb4.shape[1]])
        self.qb4_tf = tf.placeholder(tf.float32, shape=[None, self.qb4.shape[1]])

        self.u0_tf = tf.placeholder(tf.float32, shape=[None, self.u0.shape[1]])
        self.u1_tf = tf.placeholder(tf.float32, shape=[None, self.u1.shape[1]])
        self.u2_tf = tf.placeholder(tf.float32, shape=[None, self.u2.shape[1]])
        self.u3_tf = tf.placeholder(tf.float32, shape=[None, self.u3.shape[1]])
        self.u4_tf = tf.placeholder(tf.float32, shape=[None, self.u4.shape[1]])
        
        self.x_f_tf = tf.placeholder(tf.float32, shape=[None, self.x_f.shape[1]])
        self.y_f_tf = tf.placeholder(tf.float32, shape=[None, self.y_f.shape[1]])
        self.t_f_tf = tf.placeholder(tf.float32, shape=[None, self.t_f.shape[1]])
        self.q_f_tf = tf.placeholder(tf.float32, shape=[None, self.q_f.shape[1]])

        self.u0_pred, _ = self.net_u(self.x0_tf, self.y0_tf, self.t0_tf, self.q0_tf)
        self.u1_pred, _ = self.net_u(self.xb1_tf, self.yb1_tf, self.tb1_tf, self.qb1_tf)
        self.u2_pred, _ = self.net_u(self.xb2_tf, self.yb2_tf, self.tb2_tf, self.qb2_tf)
        self.u3_pred, _ = self.net_u(self.xb3_tf, self.yb3_tf, self.tb3_tf, self.qb3_tf)
        self.u4_pred, _ = self.net_u(self.xb4_tf, self.yb4_tf, self.tb4_tf, self.qb4_tf)
        self.f_pred = self.net_f(self.x_f_tf, self.y_f_tf, self.t_f_tf, self.q_f_tf)
        
        self.loss = (tf.reduce_mean(tf.square(self.u0_pred - self.u0_tf)) + \
                    tf.reduce_mean(tf.square(self.u1_pred-self.u1_tf)) + \
                    tf.reduce_mean(tf.square(self.u2_pred-self.u2_tf)) + \
                    tf.reduce_mean(tf.square(self.u3_pred-self.u3_tf)) + \
                    tf.reduce_mean(tf.square(self.u4_pred-self.u4_tf))) + \
                    tf.reduce_mean(tf.square(self.f_pred))
                
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, 
                                                                method = 'L-BFGS-B', 
                                                                options = {'maxiter': 80000,
                                                                           'maxfun': 80000,
                                                                           'maxcor': 50,
                                                                           'maxls': 50,
                                                                           'ftol': 1.0 * np.finfo(float).eps})

        self.optimizer_Adam = tf.train.AdamOptimizer(2e-3)
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)

        # tf session
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))

        init = tf.global_variables_initializer()
        self.sess.run(init)
        
    def initialize_NN(self, layers):        
        weights = []
        biases = []
        num_layers = len(layers) 
        for l in range(0, num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1, layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)  
        return weights, biases
        
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]        
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=tf.cast(xavier_stddev, tf.float32)))

    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1
        
        H = 2.0 * (X - self.lb)/(self.ub - self.lb) - 1.0
        for l in range(0, num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(tf.cast(H, tf.float32), W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def net_u(self, x, y, t, q):
        u = self.neural_net(tf.concat([x, y, t, q], 1), self.weights, self.biases)

        u_x = tf.gradients(u, x)[0]

        return u, u_x
    
    def net_f(self, x, y, t, q):
        arfa=-1
        beta=-1
        deta=1
        r=3
        u, u_x = self.net_u(x, y, t, q)
        u_t = tf.gradients(u, t)[0]
        u_tt = tf.gradients(u_t, t)[0]
        u_xx = tf.gradients(u_x, x)[0]
        u_xxx = tf.gradients(u_xx, x)[0]
        u_xxxx = tf.gradients(u_xxx, x)[0]
        u2_xx = 2*u_x*u_x+2*u*u_xx
        
        u_y = tf.gradients(u, y)[0]
        u_yy = tf.gradients(u_y, y)[0]
        f = u_tt-arfa*u_xx-beta*u_yy-r*u2_xx-deta*u_xxxx
        
        return f
    
    def callback(self, loss):
        print('Loss:', loss)
        
    def train(self, nIter):
        
        tf_dict = {self.x0_tf: self.x0, self.y0_tf: self.y0, self.t0_tf: self.t0, self.q0_tf: self.q0,
                   self.u0_tf: self.u0, self.u1_tf: self.u1, self.u2_tf: self.u2,
                   self.u3_tf: self.u3, self.u4_tf: self.u4,
                   self.xb1_tf: self.xb1, self.yb1_tf: self.yb1, self.tb1_tf: self.tb1, self.qb1_tf: self.qb1,
                   self.xb2_tf: self.xb2, self.yb2_tf: self.yb2, self.tb2_tf: self.tb2, self.qb2_tf: self.qb2,
                   self.xb3_tf: self.xb3, self.yb3_tf: self.yb3, self.tb3_tf: self.tb3, self.qb3_tf: self.qb3,
                   self.xb4_tf: self.xb4, self.yb4_tf: self.yb4, self.tb4_tf: self.tb4, self.qb4_tf: self.qb4,
                   self.x_f_tf: self.x_f, self.y_f_tf: self.y_f, self.t_f_tf: self.t_f, self.q_f_tf: self.q_f}

        start_time1 = time.time()
        for it in range(nIter):
            self.sess.run(self.train_op_Adam, tf_dict)

            # Print
            if it % 10 == 0:
                elapsed1 = time.time() - start_time1
                loss_value = self.sess.run(self.loss, tf_dict)
                print('It: %d, Loss: %.3e, Time: %.2f' %
                      (it, loss_value, elapsed1))
                start_time1 = time.time()

        self.optimizer.minimize(self.sess, 
                                feed_dict=tf_dict,
                                fetches=[self.loss],
                                loss_callback=self.callback)

    def predict(self, X_test):
                
        u_test = self.sess.run(self.u0_pred, {self.x0_tf: X_test[:, 0:1], self.y0_tf: X_test[:, 1:2], self.t0_tf: X_test[:, 2:3], self.q0_tf: X_test[:, 3:4]})
        
        f_test = self.sess.run(self.f_pred, {self.x_f_tf: X_test[:, 0:1], self.y_f_tf: X_test[:, 1:2], self.t_f_tf: X_test[:, 2:3], self.q_f_tf: X_test[:, 3:4]})
               
        return u_test, f_test


if __name__ == "__main__": 

    # Domain bounds
    lb = np.array([-5.0, -5.0, -0.5, -1])
    ub = np.array([5.0, 5.0, 0.5, 1])

    N_0 = 4000
    N_b = 2000
    N_f = 10000
    layers = [4, 40, 40, 40, 40, 40, 40, 1]

    data = scipy.io.loadmat('F:/BPINN/daima/q/q pinn/Data_q')

    tt = data['tnew'].flatten()[:, None]
    xx = data['xnew'].flatten()[:, None]
    yy = data['ynew'].flatten()[:, None]
    qq = data['qnew'].flatten()[:, None]
    Exact = data['unew']

    X, Y, T, Q = np.meshgrid(xx, yy, tt, qq)
    Exact = np.float32(Exact)

    X_star = np.hstack((X.flatten()[:, None], Y.flatten()[:, None], T.flatten()[:, None], Q.flatten()[:, None]))
    u_star = Exact.transpose(1,0,2,3).flatten()[:, None]
    
    t = np.ones(len(xx) * len(yy) * len(qq)) * lb[2]
    X1, Y1, Q1 = np.meshgrid(xx, yy, qq)
    X0 = np.hstack((X1.flatten()[:, None], Y1.flatten()[:, None], t.flatten()[:, None], Q1.flatten()[:, None]))
    u0 = Exact[:, :, 0, :].transpose(1,0,2).flatten()[:, None]

    # Boundary1
    x = np.ones(len(yy) * len(tt) * len(qq)) * lb[0]
    Y1, T1, Q1 = np.meshgrid(yy, tt, qq)
    X_b1 = np.hstack((x.flatten()[:, None], Y1.flatten()[:, None], T1.flatten()[:, None], Q1.flatten()[:, None]))
    u1 = Exact[0, :, :, :].transpose(1,0,2).flatten()[:, None]
    # Boundary2
    x = np.ones(len(yy) * len(tt) * len(qq)) * ub[0]
    X_b2 = np.hstack((x.flatten()[:, None], Y1.flatten()[:, None], T1.flatten()[:, None], Q1.flatten()[:, None]))
    u2 = Exact[-1, :, :, :].transpose(1,0,2).flatten()[:, None]
    # Boundary3
    y = np.ones(len(xx) * len(tt) * len(qq)) * lb[1]
    X1, T1, Q1 = np.meshgrid(xx, tt, qq)
    X_b3 = np.hstack((X1.flatten()[:, None], y.flatten()[:, None], T1.flatten()[:, None], Q1.flatten()[:, None]))
    u3 = Exact[:, 0, :, :].transpose(1,0,2).flatten()[:, None]
    # Boundary4
    y = np.ones(len(xx) * len(tt) * len(qq)) * ub[1]
    X_b4 = np.hstack((X1.flatten()[:, None], y.flatten()[:, None], T1.flatten()[:, None], Q1.flatten()[:, None]))
    u4 = Exact[:, -1, :, :].transpose(1,0,2).flatten()[:, None]

    np.random.seed(43210)
    idxf = np.random.choice(X_star.shape[0], N_f, replace=False)
    X_f = X_star[idxf, :]

    np.random.seed(1)
    idx1 = np.random.choice(X_b1.shape[0], N_b, replace=False)
    X_b1 = X_b1[idx1, :]
    u1 = u1[idx1, :]
    np.random.seed(12)
    idx2 = np.random.choice(X_b2.shape[0], N_b, replace=False)
    X_b2 = X_b2[idx2, :]
    u2 = u2[idx2, :]
    np.random.seed(123)
    idx3 = np.random.choice(X_b3.shape[0], N_b, replace=False)
    X_b3 = X_b3[idx3, :]
    u3 = u3[idx3, :]
    np.random.seed(1234)
    idx4 = np.random.choice(X_b4.shape[0], N_b, replace=False)
    X_b4 = X_b4[idx4, :]
    u4 = u4[idx4, :]
    np.random.seed(321)
    idx5 = np.random.choice(X0.shape[0], N_0, replace=False)
    X0 = X0[idx5, :]
    u0 = u0[idx5, :]

    ###########################

    model = PhysicsInformedNN(X0, u0, X_f, X_b1, u1, X_b2, u2, X_b3, u3, X_b4, u4, layers, lb, ub)

    start_time = time.time()
    model.train(5000) 
    elapsed = time.time() - start_time
    print('Training time: %.4f' % elapsed)

    
    qnew = 0
    dim1 = 0
    dim2 = 20
    dim3 = 40
    
    data1 = scipy.io.loadmat('F:/BPINN/daima/q/q pinn/Data_q0')
    u_test = data1['unew']
    
    X_0, Y_0, T_0, Q_0 = np.meshgrid(xx, yy, tt[dim1], qnew)
    X_0_test = np.hstack((X_0.flatten()[:, None], Y_0.flatten()[:, None], T_0.flatten()[:, None], Q_0.flatten()[:, None]))
    u_0_pre, _ = model.predict(X_0_test)
    u_0_star = u_test[:, :, 0].flatten()[:, None]
    error_u_0 = np.linalg.norm(u_0_star - u_0_pre, 2) / np.linalg.norm(u_0_star, 2)
    print('Error u_0: %e' % error_u_0)
    h0_pre = u_0_pre.reshape(X_0.shape[0], X_0.shape[1])
    u0_star = u_0_star.reshape(X_0.shape[0], X_0.shape[1])

    X_1, Y_1, T_1, Q_1 = np.meshgrid(xx, yy, tt[dim2], qnew)
    X_1_test = np.hstack((X_1.flatten()[:, None], Y_1.flatten()[:, None], T_1.flatten()[:, None], Q_1.flatten()[:, None]))
    u_1_pre, _ = model.predict(X_1_test)
    u_1_star = u_test[:, :, 20].flatten()[:, None]
    error_u_1 = np.linalg.norm(u_1_star - u_1_pre, 2) / np.linalg.norm(u_1_star, 2)
    print('Error u_1: %e' % error_u_1)
    h1_pre = u_1_pre.reshape(X_0.shape[0], X_0.shape[1])
    u1_star = u_1_star.reshape(X_0.shape[0], X_0.shape[1])

    X_2, Y_2, T_2, Q_2 = np.meshgrid(xx, yy, tt[dim3], qnew)
    X_2_test = np.hstack((X_2.flatten()[:, None], Y_2.flatten()[:, None], T_2.flatten()[:, None], Q_2.flatten()[:, None]))
    u_2_pre, _ = model.predict(X_2_test)
    u_2_star = u_test[:, :, 40].flatten()[:, None]
    error_u_2= np.linalg.norm(u_2_star - u_2_pre, 2) / np.linalg.norm(u_2_star, 2)
    print('Error u_2: %e' % error_u_2)
    h2_pre = u_2_pre.reshape(X_0.shape[0], X_0.shape[1])
    u2_star = u_2_star.reshape(X_0.shape[0], X_0.shape[1])
    

    XX, YY = np.meshgrid(xx, yy)

    fig = plt.figure(figsize=(12, 6))
    plt.subplots_adjust(wspace=0.5, hspace = 0.5)

    ax1 = fig.add_subplot(2,3,1, projection='3d')
    ax1.plot_surface(XX, YY, h0_pre, cmap=plt.get_cmap('rainbow'), antialiased=True)
#    ax1.contourf(XX, YY, h1, zdir = 'z', offset = 0, cmap=cm.coolwarm)   
    ax1.grid(False)
    ax1.set_xlabel(r'$x$', size=15)
    ax1.set_ylabel(r'$y$', size=15)
#    ax1.set_zlabel(r'$|A(x,y,t)|$', size=15, position=(-20, 4))
    ax1.set_zlim(0,2)
    ax1.set_xticks(np.linspace(-5,5,3))
    ax1.set_yticks(np.linspace(-5,5,3))
    ax1.set_zticks(np.linspace(0,3,4))
    ax1.set_title(r'$t = %.2f$' % (tt[dim1]), fontsize=15)
    ax1.view_init(10, 135)
    
    ax2 = fig.add_subplot(2,3,2, projection='3d')
    ax2.plot_surface(XX, YY, h1_pre, cmap=plt.get_cmap('rainbow'), antialiased=True)
#    ax1.contourf(XX, YY, h1, zdir = 'z', offset = 0, cmap=cm.coolwarm)   
    ax2.grid(False)
    ax2.set_xlabel(r'$x$', size=15)
    ax2.set_ylabel(r'$y$', size=15)
#    ax1.set_zlabel(r'$|A(x,y,t)|$', size=15, position=(-20, 4))
    ax2.set_zlim(0,2)
    ax2.set_xticks(np.linspace(-5,5,3))
    ax2.set_yticks(np.linspace(-5,5,3))
    ax2.set_zticks(np.linspace(0,3,4))
    ax2.set_title(r'$t = %.2f$' % (tt[dim2]), fontsize=15)
    ax2.view_init(10, 135)
    
    ax3 = fig.add_subplot(2,3,3, projection='3d')
    ax3.plot_surface(XX, YY, h2_pre, cmap=plt.get_cmap('rainbow'), antialiased=True)
#    ax1.contourf(XX, YY, h1, zdir = 'z', offset = 0, cmap=cm.coolwarm)   
    ax3.grid(False)
    ax3.set_xlabel(r'$x$', size=15)
    ax3.set_ylabel(r'$y$', size=15)
#    ax1.set_zlabel(r'$|A(x,y,t)|$', size=15, position=(-20, 4))
    ax3.set_zlim(0,2)
    ax3.set_xticks(np.linspace(-5,5,3))
    ax3.set_yticks(np.linspace(-5,5,3))
    ax3.set_zticks(np.linspace(0,3,4))
    ax3.set_title(r'$t = %.2f$' % (tt[dim3]), fontsize=15)
    ax3.view_init(10, 135)
    
    ax6 = fig.add_subplot(2,3,4)
#    ax6.plot(yy, h1[:, 0], linewidth=3, color = 'b', linestyle ='-')
#    ax6.plot(yy, u1[:, 0], linewidth=3, color = 'r', linestyle ='--')
    ax6.set_xticks(np.linspace(-5,5,3))
    ax6.set_yticks(np.linspace(-5,5,3))
    ax6.set_xlabel(r'$x$', size=15)
    ax6.set_ylabel(r'$y$', size=15)
    ax6.set_title(r'$t = %.2f$' % (tt[dim1]), fontsize=15)
#    ax6.legend(labels = ['Prediction','Exact'], loc='upper left', frameon=False)
#    ax6.contourf(XX,YY,h1,cmap=plt.get_cmap('rainbow'))
    g1=ax6.imshow(h0_pre,cmap=plt.get_cmap('rainbow'),extent=[-5,5,-5,5],interpolation='nearest',  
                  origin='lower')
    cbar_ax = fig.add_axes([0.33, 0.11, 0.01, 0.345])
    cmap = cm.Spectral_r
    plt.colorbar(g1, cax=cbar_ax, cmap=cmap)
    
    
    ax7 = fig.add_subplot(2,3,5)
#    ax7.plot(yy, h2[:, 0], linewidth=3, color = 'b', linestyle ='-')
#    ax7.plot(yy, u2[:, 0], linewidth=3, color = 'r', linestyle ='--')
    ax7.set_xticks(np.linspace(-5,5,3))
    ax7.set_yticks(np.linspace(-5,5,3))
    ax7.set_xlabel(r'$x$', size=15)
    ax7.set_ylabel(r'$y$', size=15)
    ax7.set_title(r'$t = %.2f$' % (tt[dim2]), fontsize=15)
    g2=ax7.imshow(h1_pre,cmap=plt.get_cmap('rainbow'),extent=[-5,5,-5,5],interpolation='nearest',  
                  origin='lower')
    cbar_ax = fig.add_axes([0.605, 0.11, 0.01, 0.345])
    cmap = cm.Spectral_r
    plt.colorbar(g2, cax=cbar_ax, cmap=cmap)

    ax8 = fig.add_subplot(2,3,6)
#    ax8.plot(yy, h3[:, 0], linewidth=3, color = 'b', linestyle ='-')
#    ax8.plot(yy, u3[:, 0], linewidth=3, color = 'r', linestyle ='--')
    ax8.set_xticks(np.linspace(-5,5,3))
    ax8.set_yticks(np.linspace(-5,5,3))
    ax8.set_xlabel(r'$x$', size=15)
    ax8.set_ylabel(r'$y$', size=15)
    ax8.set_title(r'$t = %.2f$' % (tt[dim3]), fontsize=15)
    g3=ax8.imshow(h2_pre,cmap=plt.get_cmap('rainbow'),extent=[-5,5,-5,5],interpolation='nearest',  
                  origin='lower')
    cbar_ax = fig.add_axes([0.88, 0.11, 0.01, 0.345])
    cmap = cm.Spectral_r
    plt.colorbar(g3, cax=cbar_ax, cmap=cmap)

    # plt.savefig('H:/q/q0.pdf')
