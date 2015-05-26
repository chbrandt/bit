import numpy as np

class Simulator:
    @staticmethod
    def gaussian(img, stdev=3.):
      """ Generate zero-mean additive gaussian noise
    
      Input:
       - img <ndarray>
       - stdev <float>
    
      Output:
       <ndarray>
      """
    
      noiz = np.random.normal(0., stdev, img.shape)
      noisy_img = (1. * img) + noiz
      return noisy_img
    
    @staticmethod
    def poisson(img):
        """
        Add poisson noise to image array
    
        Notice that only non-zero pixels will be uptaded with noise effect.
    
        Input:
         - image  :ndarray: Image array [float, 2-dim array]
    	
        Output:
         - noisy  :ndarray: Given imagem with Poisson values
        ---
        """
    
        img_nzy = np.random.poisson(img).astype(float);
        return img_nzy;
    
    @staticmethod
    def salt_n_pepper(img, perc=10):
        """ (EPD webinar)
        Generate salt-and-pepper noise in an image. 
        Salt-and-Pepper noise is defined as randomly dispersed values of 0 and 255.
    
        Input:
        - img
        - perc
    
        Output:
        - <ndarray>
        """
    
        # Create a flat copy of the image
        flat = img.ravel().copy
    
        # The total number of pixels
        total = len(flat)
    
        # The number of pixels we need to modify
        nmod = int(total * (perc/100.))
    
        # Random indices to modify
        indices = np.random.random_integers(0, total-1, (nmod,))
    
        # Set the first half of the pixels to 0
        flat[indices[:nmod/2]] = 0
    
        # Set the second half of the pixels to 255
        flat[indices[nmod/2:]] = 255
    
        return flat.reshape(img.shape)

