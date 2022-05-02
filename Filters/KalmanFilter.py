class KalmanFilter(object):
    def __init__(self, init_pos, max_angle_change,
                 dt):
        """Initialize"""
        self.x_pos = init_pos['x']
        self.y_pos = init_pos['y']
        self.max_angle_change = max_angle_change
        self.dt = dt
        """ X = [x, y, x', y', acc]^T
        x = x + x't + S*acc*t^2
        y = y + y't + 1*acc*t^2
        x' = x' + S*acc*t
        y' = y' + 1*acc*t
        acc = acc
        """
        self.F = np.matrix([[1,0,dt, 0,self.S_acc*(dt**2)],
                           [0,1, 0,dt,           (dt**2)],
                           [0,0, 1, 0,     self.S_acc*dt],
                           [0,0, 0, 1,                dt],
                           [0,0, 0, 0,                 1]
        ])  # 5 by 5
        self.H = np.matrix([[1,0,0,0,0],
                           [0,1,0,0,0]]) # 2 by 5
        self.Q = np.diag([0, 0, 0, 0, 0])    # 5 by 5
        self.R = np.diag([0.05**2,0.05**2])    # 2 by 2
        self.I = np.identity(5)
        self.target_predictions = dict([])
        self.target_confidences = dict([])
        self.obs_id = set()

    def Initialize_new_target(self,i,pos_x,pos_y):
        X = np.matrix([[pos_x],[pos_y],[.0],[.0],[.0]])  # 5 by 1
        P = np.diag([1/3, 1/3, 1/2, 1/2, 1])  # 5 by 5
        self.target_predictions[i] = X
        self.target_confidences[i] = P
        pass
        
    def run_kf(self,i,meteorite_x_obs, meteorite_y_obs):
        Z  = np.matrix([[meteorite_x_obs],[meteorite_y_obs]]) # the obs vector, which size is 2 by 1
        X = self.target_predictions[i]
        P = self.target_confidences[i]
        H = self.H
        R = self.R
        F = self.F
        Q = self.Q
        # Prediction
        X_h = F * X
        P_h = F * P * F.T + Q
        # Observation
        K = P_h * H.T * (np.linalg.pinv(H*P_h*H.T + R))
        self.target_predictions[i] = X_h + K * (Z - H * X_h)
        self.target_confidences[i] = (self.I - K * H) * P_h
        return self.target_predictions[i].item(0,0),self.target_predictions[i].item(1,0)



    
