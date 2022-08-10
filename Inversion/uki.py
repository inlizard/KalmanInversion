## UKI  
## Longlong Wang 
## Institute of Geology and Geophysics, Chinese Academy of Sciences, China
## Thu Mar 31 19:59:32 CST 2022

import numpy as np

class UKI:

    def __init__(self, theta_names, theta0_mean, thetata0_cov, y, sigma_ita, alpha_reg, update_freq, update_r = 0, modified_unscented_transform=0, multiple=2):
        N_theta = theta0_mean.shape[0]
        N_y = y.shape[0]

        # ensemble size
        N_ens = 2 * N_theta + 1

        c_weights = np.zeros(N_theta)
        mean_weights = np.zeros(N_ens)
        cov_weights = np.zeros(N_ens)

        kappa = 0.0
        beta = 2.0
        alpha = np.min([np.sqrt(4/(N_theta + kappa)), 1])
        lam = alpha * alpha * (N_theta+kappa) - N_theta

        c_weights[:] = np.sqrt(N_theta + lam)
        mean_weights[0] = lam/(N_theta + lam)
        mean_weights[1:] = 1/(2*(N_theta + lam))
        cov_weights[0] = lam/(N_theta + lam) + 1 - alpha * alpha + beta
        cov_weights[1:] = 1/(2*(N_theta + lam))

        if modified_unscented_transform:
            mean_weights[0] = 1.0
            mean_weights[1:] = 0.0

        theta_mean = []
        theta_mean.append(theta0_mean)
        thetata_cov = []
        thetata_cov.append(thetata0_cov)

        y_pred = []
        sigma_w = (2 - alpha_reg*alpha_reg) * thetata0_cov
        sigma_miu = 2 * sigma_ita
        r = theta0_mean
        iters = 0
    
        self.multiple = multiple
        self.thetanames = theta_names
        self.theta_means = theta_mean # change
        self.thetata_covs = thetata_cov # change
        self.y_preds = y_pred
        self.y = y
        self.sigma_ita = sigma_ita
        self.N_ens = N_ens # change
        self.N_theta = N_theta #change
        self.N_y = N_y
        self.c_weights = c_weights
        self.mean_weights = mean_weights
        self.cov_weights = cov_weights
        self.sigma_w = sigma_w
        self.sigma_miu = sigma_miu
        self.alpha_reg = alpha_reg
        self.r = r
        self.update_freq = update_freq
        self.iter = iters
        self.update_r = update_r
        self.cut = []

    def construct_sigma_ensemble(self, x_mean, x_cov):

        N_x = x_mean.shape[0]
        c_weights = self.c_weights
        chol_xx_cov = np.linalg.cholesky(x_cov.T)
        
        ## clip the minimal singular values t0 1e-8

        try:
            chol_xx_cov = np.linalg.cholesky(x_cov)
        except:
            s1,v1,d1 = np.linalg.svd(x_cov)
            v1 = np.clip(v1,a_min = 1e-8,a_max = None)
            q1 = np.linalg.qr(np.sqrt(v1)[:,None] * d1)[-1]
            chol_xx_cov = q1.T

        x = np.zeros((2*N_x+1, N_x))
        
        ##  1 
        # x[0, :] = x_mean
        # for i in range(N_x):
        #     x[i+1, :] = c_weights[i] * chol_xx_cov[:, i]
        #     x[i+1+N_x, :] = - c_weights[i] * chol_xx_cov[:, i]
        
        ##  2
        x1 = np.zeros((N_x,N_x))
        x1 = c_weights * chol_xx_cov
        x2 = np.hstack((x1,-x1))
        x[1:,:] = x2.T
        x += x_mean

        return x
    
    def construct_mean(self, x):
        """
        x_mean:N_x
        """
        mean_weights = self.mean_weights
        N_ens, N_x = x.shape
        x_mean = np.zeros(N_x,)
        x_mean = mean_weights @ x

        return x_mean

    def construct_cov(self, x, x_mean):
        """
        xx_cov :N_x * N_x
        """
        cov_weights = self.cov_weights

        N_ens, N_x = self.N_ens, x_mean.shape[0]
        N_ens = self.N_ens

        xx_cov = np.zeros((N_x, N_x))
        dX = x - x_mean  # N_ens * N_x
        xx_cov = cov_weights * dX.T @ dX
        return xx_cov

    def construct_cov2(self, x, x_mean, y, y_mean):
        """ 
        xy_cov :N_x * N_y
        """
        N_ens, N_x, N_y = self.N_ens, x_mean.shape[0], y_mean.shape[0]
        cov_weights = self.cov_weights

        xy_cov = np.zeros((N_x, N_y))
        dX = x - x_mean # N_ens * N_x
        dY = y - y_mean # N_ens * N_y
        xy_cov = cov_weights * dX.T @ dY  # N_x * N_y
        return xy_cov

    def update_ensemble(self, forward):

        multiple = self.multiple
        self.iter += 1
        if self.update_freq > 0 and (self.iter % self.update_freq==0):
            self.sigma_w = (multiple - self.alpha_reg*self.alpha_reg) * self.thetata_covs[-1]
        if self.update_r > 0 and (self.iter % self.update_r == 0):
            self.r = self.theta_means[-1]
            
        theta_mean = self.theta_means[-1]
        thetata_cov = self.thetata_covs[-1]
        y = self.y

        alpha_reg = self.alpha_reg
        r = self.r
        sigma_w = self.sigma_w
        sigma_miu = self.sigma_miu

        N_theta, N_y, N_ens = self.N_theta, self.N_y, self.N_ens

        # prediction step
        theta_p_mean = alpha_reg * theta_mean + (1-alpha_reg) * r
        thetata_p_cov = alpha_reg*alpha_reg * thetata_cov + sigma_w

        # Generate sigma points
        theta_p = self.construct_sigma_ensemble(
            theta_p_mean, thetata_p_cov)  # -> a*2+1,a
        thetata_p_cov = self.construct_cov(theta_p, theta_p_mean)

        # Analysis step
        g = np.zeros((N_ens, N_y))*1.0
        for i in range(N_ens):
            g[i] = forward(theta_p[i])
        
        ## Y
        g_mean = self.construct_mean(g)  # N_data
        gg_cov = self.construct_cov(g, g_mean) + sigma_miu 
        tag_cov = self.construct_cov2(
            theta_p, theta_p_mean, g, g_mean)  # a * b

        # if len(gg_cov) >= 1:
        g_ = np.linalg.inv(gg_cov)  # N_y * N_y
        tmp = tag_cov @ g_  # N_x * N_y @ N_y * N_y -> N_x*N_y
        # N_params
        theta_mean = theta_p_mean + tmp @ (y-g_mean)  # N_x + N_x*N_y @ N_y*1 -> N_x*1
        # N_data
        thetata_cov = thetata_p_cov - tmp @ (tag_cov.T)  # a * a
        # else:
        #     tmp = tag_cov / (gg_cov)
        #     theta_mean = theta_p_mean + tmp @ (y-g_mean)
        #     thetata_cov = thetata_p_cov - tmp @ (tag_cov.T)

        self.y_preds.append(g_mean)
        self.theta_means.append(theta_mean)
        self.thetata_covs.append(thetata_cov)

    def UKI_run(self, N_iter, forward,cut_open=False,change_var=False,min_thk = 0.2,show_error=1):

        for i in range(N_iter):
            self.update_ensemble(forward)
            tmp = 0.5*(self.y_preds[i]-self.y)
            error1 = tmp[None, :] @ self.sigma_ita@ tmp[:, None]
            error1 = error1[0][0]
            if i%show_error == 0:
                print("optimization error at iter %d = %s\n" % (i+1, error1))
            # print("Frobenius norm of the covariance at iter %d = %s" %
            #       (i, np.linalg.norm(self.thetata_covs[i])))
            cuts = self.cut
            
            self.cut.append(error1)
            if cut_open and i>1:
                cut1 = cuts[-1]
                if error1 > cut1 and error1<0.0001:
                    print("UKI maybe converged：%s\n iter:%d"%(error1,i+1))
                    break

if __name__ == "__main__":

    def forward(theta):
        return G @ theta

    alpha_reg = 1
    update_freq = 0 
    N_iter = 20 
    
    theta_ref = np.array([1/3,17/12])
    G = np.array([1,2,3,4,5,6]).reshape(3,2) 
    y = np.array([3,7,10]) 

    sigma_ita = np.eye(len(y))*0.01
    N_theta = len(theta_ref)
    N_y = len(y)
    theta0_mean = np.zeros((N_theta))
    thetata0_cov = np.eye(N_theta) * 0.25 
    theta_names = "abc"
    modified_unscented_transform = 0
    update_freq = 0

    uki_obj = UKI(theta_names,theta0_mean,thetata0_cov,y,sigma_ita,alpha_reg,update_freq,modified_unscented_transform)
    uki_obj.update_ensemble(forward)
    uki_obj.UKI_run(10,forward)
