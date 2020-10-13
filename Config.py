class Config:
    """Ana config sınıfı"""

    def __init__(self):
        ## alt config sınıflarının örneklendirilmesi
        #self.preprocess_conf = self.preprocessConf()

    def reveal(self):
        # calling the 'Inner' class function display
        #self.inner.inner_display("Calling Inner class function from Outer class")

    class FeatureExtractionConf:
        """Preproccess Config """
        sampling_rate = None
        duration = 2
        hop_length = 347 * duration  # to make time steps 128
        fmin = 20
        fmax = sampling_rate // 2
        n_mels = 128
        n_fft = n_mels * 20
        samples = sampling_rate * duration

