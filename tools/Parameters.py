import configparser


class Parameters:
    def __init__(self, ini):
        config = configparser.ConfigParser()
        config.read(ini)
        params = config['DEFAULT']
        self.name = params['name']
        self.iplot = params.getint('iplot')
        self.saveplot = params.getint('saveplot')
        self.denoise = params.getint('denoise')
        self.decimate = params.getint('decimate')
        self.minima = params.getint('minima')
        self.rot_detrend = params.getint('rot_detrend')
        self.clean = params.getint('clean')
        self.grid_by_number = params.getint('grid_by_number')
        self.save_granulo = params.getint('save_granulo')
        self.save_grain = params.getint('save_grain')
        self.res = params.getfloat('res')
        self.n_scale = params.getint('n_scale')
        self.min_scale = params.getfloat('min_scale')
        self.max_scale = params.getfloat('max_scale')
        self.knn = params.getint('knn')
        self.rad_factor = params.getfloat('rad_factor')
        self.max_angle1 = params.getfloat('max_angle1')
        self.max_angle2 = params.getfloat('max_angle2')
        self.min_flatness = params.getfloat('min_flatness')
        self.fit_method = params['fit_method']
        self.a_quality_thresh = params.getfloat('a_quality_thresh')
        self.min_diam = params.getfloat('min_diam')
        self.n_axis = params.getint('n_axis')
        self.n_min = params.getint('n_min')
        self.dx_gbn = params.getfloat('dx_gbn')
