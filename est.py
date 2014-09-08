# -*- coding: utf-8 -*-
#-----------------------------------------------------------------------------
# Name:        est.py
# Author:      Yandi LI
# Created:     2014/5/12
# Objective:   Factorization Machine model with Bayesian Personalized Ranking 
# Class:       FM(), Sampler(), MixedSampler(), UniformUserUniformItem()
#-----------------------------------------------------------------------------
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_random_state
from sklearn.cross_validation import KFold
from scipy.sparse import lil_matrix
from collections import Counter

np.seterr(invalid='raise', under='ignore') # raise error if NaN is encountered

class FM(BaseEstimator, ClassifierMixin):

  def __init__(self, 
              K = 20, # number of factors
              learning_rate=0.05,
              bias_regularization=0.1,   # lambda for w^I_{i} and w^T_{tid}
              context_regularization_multiplier = 0.01, # lambda for U^C_{k,f} for all contextual factors
              positive_item_regularization=0.0025, # lambda for U^I_{i}
              negative_item_regularization=0.005, # lambda for U^I_{j}
              type_regularization = 0.1, # lambda for U^T_{tid}
              update_negative_item_factors=True,
              use_type = True, # whether to include venue type in the model
              num_vld_samples = 500,
              random_state=None):
      self.K = K
      self.learning_rate = learning_rate
      self.bias_regularization = bias_regularization
      self.context_regularization_multiplier = context_regularization_multiplier
      self.type_regularization = type_regularization
      self.positive_item_regularization = positive_item_regularization
      self.negative_item_regularization = negative_item_regularization
      self.update_negative_item_factors = update_negative_item_factors
      self.use_type = use_type
      self.num_vld_samples = num_vld_samples
      self.random_state = random_state


  def init(self, data):
      """ initialize the class with a data object
      @Parameters:
      ----------------------------------
      | data: {'c_attr': num array, 'triple': scipy.sparse.csr_matrix , 'v_attr': num array}
        - 'c_attr': each row is a tuple (cid, feature_1, ..., feature_n_c_attr)
        - 'v_attr': each row is a tuple (vid, venue_type)
        - 'triple': (cid, vid, cnt). The number of checkins cnt on venue vid in context cid 
      """
      self.data = data             
      rds = check_random_state(self.random_state)

      #=========  descriptive parameters  ==========#
      # number of context features 
      self.n_c_attr = self.data['c_attr'].shape[1]
      # number of discrete factors in each context feature dimension, use the maximal index as our guess
      self.n_c_attr_f = {i:max(self.data['c_attr'][:,i])+1 for i in xrange(self.n_c_attr)} 
      # cardinality of contexts
      self.n_context_e = self.data['triple'].shape[0] # observed contexts,  in Y+
      # number of venues (generally denoted as item) 
      self.n_item = self.data['triple'].shape[1]
      if self.use_type:
        # number of item attribute dimensions, in our case there is only 1, i.e., type
        self.n_i_attr = 1
        # number of discrete factors in item attribute 1, i.e. number of venue types 
        self.n_i_attr_f = max(self.data['v_attr'][:,0])+1
      
      #=========  initialize FM model parameters  ==========#
      # cummulative(context-agnostic) popularity scores of venues
      self.ppl = self.popularity(self.data['triple'])

      # initialize item bias by their cummulative score to boost training speed
      # adds a little variation in order not to get trapped in the beginning
      #~~ self.item_bias = np.zeros(self.n_item) # w^I_{vid}
      self.item_bias = np.array(self.ppl) + rds.randn(self.n_item)*0.0001 
      
      # initialize latent parameters to be uniformly distributed around 0
      self.context_factors = {a:rds.random_sample((self.n_c_attr_f[a], self.K))-0.5 \
          for a in xrange(self.n_c_attr)}  # U^A_{a}{i}

      self.item_factors = rds.random_sample((self.n_item, self.K))-0.5 # U^V_{vid}
      
      if self.use_type:
        self.type_bias = np.zeros(self.n_i_attr_f) # w^T_{tid}
        self.type_factors = rds.random_sample((self.n_i_attr_f, self.K))-0.5 # U^T_{tid}

      self.context_regularization = {i:self.context_regularization_multiplier
          for i in xrange(self.n_c_attr)}

      return self


  def summary(self,data):
      # number of context attribute dimensions 
      n_c_attr = data['c_attr'].shape[1]
      # number of discrete factors in each context attribute dimension, use the maximal index as our guess
      n_c_attr_f = {i:max(data['c_attr'][:,i])+1 for i in xrange(n_c_attr)} 
      # cardinality of contexts
      n_context_e = data['triple'].shape[0] # observed contexts,  in Y+
      n_context_t = np.prod(n_c_attr_f.values()) # theoretical cardinality
      # cardinality of venues (generally denoted as item) 
      n_item = data['triple'].shape[1]
      # number of item attribute dimensions, in our case there is only 1, i.e., type
      n_i_attr = 1
      # number of discrete factors in item attribute 1, i.e. number of venue types 
      n_i_attr_f = max(data['v_attr'][:,0])+1
      
      print "Use item type information:{0}".format(self.use_type)
      print "{0} context dimensions".format(n_c_attr)
      print "Number of factors in each context dimension: {0}".format(n_c_attr_f)
      print "{0} item types".format(n_i_attr_f)
      print """{0} items have been observed, {1} out of
          {2} context combinations have been observed.""".format(\
            n_item, n_context_e, n_context_t)
      print "Sparsity of the context-item matrix:",\
            100-100.0*data['triple'].nnz/n_context_e/n_item
      # number of context per observed item
      print 'context per observed item ',\
            np.percentile(Counter(data['triple'].nonzero()[1]).values(),[5,25,50,75,95])
      # number of items per observed context
      print 'items per observed context: ',\
            np.percentile(Counter(data['triple'].nonzero()[0]).values(),[5,25,50,75,95])
      #checkins per line
      print 'checkins per item: ',\
            np.percentile(data['triple'].sum(axis=0),[5,25,50,75,95])
      #checkins per observed line
      print 'checkins per observed item: ',\
            np.percentile([i for i in data['triple'].sum(axis=0).flat if i>0],[5,25,50,75,95])
      #checkins per obs.context and obs.line
      print 'checkins per observed context and observed item: ',\
            np.percentile(data['triple'].data,[5,25,50,75,95])


  def create_validation_samples(self):
      """ create a sample of (cid, i, j) of self.num_vld_samples to 
      evaluate training accuracy
      >>> model.sampler.init(data['triple']) #doctest:+ELLIPSIS  
      Random State reset...
      >>> model.create_validation_samples()[:5]
      sampling 500 <context,item i,item j> triples...
      [(226, 4359, 5631), (335, 1042, 7300), (237, 595, 1243), (236, 2386, 5713), (336, 5313, 5102)]
      """
      # apply rule of thumb to decide the size of samples over which to compute loss
      if self.num_vld_samples == None:
        self.num_vld_samples = int(100*self.n_context_e**0.5)

      print 'sampling {0} <context,item i,item j> triples...'.format(self.num_vld_samples)
      # generate a list of (context array, positive itemid, negative itemid)
      vld_samples = list(self.sampler.generate_samples(self.num_vld_samples)) 
      return vld_samples
      

  def cross_validation_split(self, n_folds=5, n_neg=3):
      r""" split observation into training and test into k folds,
      generate k folds test_samples and the mask matrix
      @Parameters:
      ------------------------------------------
      | n_folds: int, number of folds in cross validation
      | n_neg:   int, number of negative items against one positve observation in generating the loss sample
      @Return:
      ------------------------------------------
      generator:
      | test_samples:  list of triplet, (u,i,j) list 
                      where item i is preferred to j by u
      | test_mask: linked in list sparse matrix, boolean type, scipy.sparse.lil_matrix
                    a mask matrix used to inform the sampler that 
                    when test_mask[u,i] == True, neither (u,i,j) nor (u,j,i) 
                    should be sampled for training. This splits up the dataset, 
                    to prevent the model from memorizing.
      >>> model.sampler.init(data['triple']) #doctest:+ELLIPSIS
      Random State reset...
      >>> test_samples, test_mask = model.cross_validation_split(n_folds=5,n_neg=3).next()
      >>> test_samples[:6]
      [(0, 153, 7008), (0, 153, 7343), (0, 153, 5900), (0, 246, 1042), (0, 246, 7300), (0, 246, 7340)]
      >>> test_mask[0,153] and test_mask[0,7008] and test_mask[0,7343]
      True
      """
      # sparse matrix with 1 when (u,i) is in test and kept from fitting
      data = self.data['triple']
      sampler = self.sampler
      kf = KFold(data.size, n_folds=n_folds, shuffle=True, random_state=self.random_state)
      U,I = data.nonzero()
      for _, test_positive in kf:
        test_samples = []
        test_mask = lil_matrix(data.shape, dtype='bool') 
        for idx in test_positive:
          test_mask[U[idx], I[idx]] = True
          for i in xrange(n_neg):
            j = sampler.sample_negative_item(data[U[idx]].indices)
            test_samples.append((U[idx], I[idx], j))
            test_mask[U[idx], j] = True 
        yield test_samples, test_mask  
       
  
  def fit(self, data, num_iters=10, mode='cv', n_folds=5, stop=1, n_neg=3):
      """ compute the cross-validated pairwise accuracy 
      @Parameters:
      --------------------------------------------
      | data: dict, {'c_attr','triple','v_attr'}, full dataset
      | n_folds: int, number of folds in cross validation
      | stop: int, terminate after how many folds of cross validation 
      | n_neg: int, number of negative items against one positve 
              observation when generating the loss samples
      | n_iters: int, max iterations of training
      | mode: 'cv': cross_validation to test model performance
                    when stop=1, it is equivalent to train-test split
                    when stop=n_folds, full K-folds performance
              'fit': fit the model with all data, the output test error should
                    be considered as training error
      @Return:
      --------------------------------------------
      | scores: list of test AUC score of CV folds
      """
      self.init(data)   # init model
      self.summary(data)
      # initialize sampler here, so that randomizer is reset 
      # every time fit() is called
      self.sampler = MixedSampler(True, self.random_state, 0.4)
      self.sampler.init(self.data['triple']) 
      
      scores = []
      for i, (test_samples, test_mask) in enumerate(
          self.cross_validation_split(n_folds, n_neg)):
        ppl = self.popularity(self.data['triple'])
        print 'most_popular_auc:', self.uncontextual_auc(ppl,test_samples)
        
        if i+1 > stop: break
        print '==================================================='
        print 'Cross Validation: Fold = %d'%(i+1)
        print '==================================================='
        self.init(data)   # init model
        if mode == 'fit':
          # dummy mask, use the whole data for training
          self.sampler.set_mask(lil_matrix(data['triple'].shape, dtype='bool'))
        elif mode == 'cv':
          self.sampler.set_mask(test_mask)
        self.test_samples = test_samples
        self.__fit__(num_iters)
        scores.append(self.score(test_samples))

      return scores  
    

  def __fit__(self, num_iters):
      """ one SGD iteration training based on sampled list of triplet (cid,i,j)
      """
      # bootstrap samples to calculate training loss
      vld_samples = self.create_validation_samples()
      print 'initial train = {0}, test = {1}'.format(
          self.score(vld_samples), self.score(self.test_samples))
    
      learning_rate = self.learning_rate
      
      for it in xrange(num_iters):
        # decay learning rate every 30 iterations
        if (it+1)%30 == 0:
          learning_rate = self.learning_rate/((it+1)/30+1)
          print 'learning_rate', learning_rate

        # in each iteration, sample 500 pairs and update model parameters
        # test set will not be sampled as we've added a mask to the sampler
        for cid,i,j in self.sampler.generate_samples(500):
          self.update_factors(cid,i,j,learning_rate)
        print 'iteration {0}: train = {1}, test = {2}'.format(
            it+1, self.score(vld_samples), self.score(self.test_samples))
      
      return self


  def predict(self,context,i):
      """ score prediction of venue i in context
      @Parameters:
      ----------------------------------
      | context: 1d-array, context features
      | i: int, venue id
      >>> model.data['c_attr'][1,1:] # context features, hour=8, dow=5, weather=1
      array([8, 5, 1], dtype=int64)
      >>> model.predict(data['c_attr'][1,1:],5) # predicted score of venue 5 in context 1
      0.65172777511844859
      """
      if self.use_type:
        tid = self.data['v_attr'][i,0]
        score = self.item_bias[i] + self.type_bias[tid]
        for k,c in enumerate(context):
          score += np.dot(self.context_factors[k][c], 
              self.item_factors[i] + self.type_factors[tid])
      else:
        score = self.item_bias[i] 
        for k,c in enumerate(context):
          score += np.dot(self.context_factors[k][c], self.item_factors[i])

      if np.isnan(score) or np.isinf(score):
        raise OverflowError
      return score


  def rate_all(self,context):
      """ scoring all venues in a context. Return a 1d-array
      @Paramters:
      -------------------------------------
      | context: 1d-array, context features
      >>> model.rate_all(data['c_attr'][1,1:])
      array([  0.95935873,   0.12649043,  11.60256552, ...,   5.14625727,
               1.89909029,   8.43583068])
      """
      y_ = []
      for i in xrange(self.n_item):
        score = self.predict(context, i)
        y_.append(score)
      return np.array(y_)


  def score(self, loss_samples):
      """ AUC of the pairwise ordering of the model prediction    
      >>> model.score(loss_samples=[(4,1107,7156), (5,2,1), (10,12,25)])
      0.7698066103354404
      """
      ranking_loss = 0;
      for cid,i,j in loss_samples:
        contexts = self.data['c_attr'][cid,:]
        x = self.predict(contexts,i) - self.predict(contexts,j)
        # avoid overflow 
        if x < 100: 
          ranking_loss += 1.0/(1.0+np.exp(x))
      return 1.0 - 1.0*ranking_loss/len(loss_samples)


  def update_factors(self, cid, i, j, 
                    learning_rate, update_item = True, update_context = True):
      """apply SGD update given one sample of (cid,i,j) tuple"""
      contexts = self.data['c_attr'][cid,:]
      
      if self.use_type:
        ti = self.data['v_attr'][i,0]
        tj = self.data['v_attr'][j,0]
        update_type = ti != tj # if type(i)==type(j), no update for type factors
      else:
        update_type = False

      x = self.predict(contexts,i) - self.predict(contexts,j)
      if x > 100: x = 100 # avoid overflow, if z = 0, there is no update
      z = 1.0/(1.0+np.exp(x))
      
      # item bias and item factors
      if update_item:
        g = 1.0
        d = z * g - self.bias_regularization * self.item_bias[i] 
        self.item_bias[i] += learning_rate * d
        d = -z * g - self.bias_regularization * self.item_bias[j]
        self.item_bias[j] += learning_rate * d
      
        g = sum(self.context_factors[k][c] for k,c in enumerate(contexts))
        d = z * g - self.positive_item_regularization * self.item_factors[i]
        self.item_factors[i] += learning_rate * d
        d = -z * g - self.negative_item_regularization * self.item_factors[j]
        self.item_factors[j] += learning_rate * d
      
      # context factors
      if update_context:
        if not update_type:
          g = self.item_factors[i] - self.item_factors[j]
        else:
          g = self.item_factors[i] - self.item_factors[j]+\
              self.type_factors[ti] - self.type_factors[tj]
        for k, c in enumerate(contexts):
          d = z * g - self.context_regularization[k] * self.context_factors[k][c]
          self.context_factors[k][c] += learning_rate * d

      # type bias and type factors
      if update_type:
        g = 1.0
        d = z * g - self.bias_regularization * self.type_bias[ti]
        self.type_bias[ti] += learning_rate * d
        d = -z * g - self.bias_regularization * self.type_bias[tj]
        self.type_bias[tj] += learning_rate * d

        g = sum(self.context_factors[k][c] for k,c in enumerate(contexts))
        d = z * g - self.type_regularization * self.type_factors[ti]
        self.type_factors[ti] += learning_rate * d
        d = -z * g - self.type_regularization * self.type_factors[tj]
        self.type_factors[tj] += learning_rate * d


  @staticmethod
  def popularity(data, idx=None):
      """ return the cummulative #checkins on venues summed over a list of context id 
      Parameters:
      ---------------------------------
      | data: triplet, sparse.matrix
      | idx: list of context id to consider, used for cross-validation
      >>> from scipy.sparse import csr_matrix
      >>> FM.popularity(csr_matrix(np.array([[1,2,2],[0,1,2]])))
      [1, 3, 4]
      """
      n_context, n_venue = data.shape
      if idx == None:
        idx = range(n_context)
      ppl = Counter()
      for v,c in zip(data[idx].nonzero()[1], data[idx].data):
        ppl[v] += c
      return [ppl[i] for i in xrange(n_venue)]


  @staticmethod
  def uncontextual_auc(ppl, loss_samples):
      """ calculate pairwise ordering accuracy of a list of predicted venue scores, 
      compared against a list of (cid,i,j) tuples, containing the truth that 
      venue i has higher score than j in context cid
      Parameters:
      ----------------------------------
      | ppl: a list of numbers as predicted venue scores
      | loss_samples: list of (cid,i,j) tuples as a test sample of ordering accuracy 
      >>> FM.uncontextual_auc([1,2,4,3],[(1,0,2),(2,2,1)])
      0.5
      """
      loss = 0
      for _, i, j in loss_samples:
        if ppl[i] > ppl[j]: pass
        elif ppl[i] == ppl[j]: loss += 0.5
        else: loss += 1
      return 1.0-1.0*loss/len(loss_samples)



# sampling strategies

class Sampler(object):

    def __init__(self,sample_negative_items_empirically, random_state=None):
        self.sample_negative_items_empirically = sample_negative_items_empirically
        self.random_state = random_state

    def init(self, data, max_samples=None, mask=None):
        """
        @Parameters:
        ------------------------------
        | data: scipy.sparse.csr_matrix, triplets of (cid,vid,cnt)
        | max_samples: int, number of samples to generate for each call
        | mask: scipy.sparse.lil_matrix, a matrix indicates which part of data 
                are not used for sampling, to respect a training-test split.
                If (cid,vid) == True, then neither (_,vid,i) or (_,j,vid) 
                will be sampled
        """
        self.data = data
        self.num_users, self.num_items = data.shape
        self.max_samples = max_samples
        self.rds = check_random_state(self.random_state)
        print 'Random State reset'
        self.mask = lil_matrix(data.shape,dtype='bool') if mask == None else mask
        return self
        
    def set_mask(self, mask):
        self.mask = mask
        return self
    
    def add_mask(self, mask):
        self.mask = (self.mask + mask).astype('bool')
        return self

    def sample_user(self):
        u = self.uniform_user()
        num_items = self.data[u].getnnz()
        assert(num_items > 0 and num_items != self.num_items)
        return u

    def sample_negative_item(self,user_items):
        j = self.random_item()
        while j in user_items:
            j = self.random_item()
        return j

    def uniform_user(self):
        return self.rds.randint(0,self.num_users-1)

    def random_item(self):
        """sample an item uniformly or from the empirical distribution
           observed in the training data
        """
        if self.sample_negative_items_empirically:
            # just pick something someone rated!
            u = self.uniform_user()
            i = self.rds.choice(self.data[u].indices)
        else:
            i = self.rds.randint(0, self.num_items-1)
        return i

    def num_samples(self,n):
        if self.max_samples is None:
            return n
        return min(n,self.max_samples)


class UniformUserUniformItem(Sampler):

    def generate_samples(self, max_samples=None):
        self.max_samples = max_samples
        for _ in xrange(self.num_samples(self.data.nnz)):
          u = self.uniform_user()
          # sample positive item
          i = self.rds.choice(self.data[u].indices)
          while self.mask[u,i] == True:
            u = self.uniform_user()
            i = self.rds.choice(self.data[u].indices)
          j = self.random_item()
          while self.mask[u,j] == True or self.data[u,i] == self.data[u,j]:
            j = self.random_item()
          if self.data[u,i] > self.data[u,j]:
            yield u,i,j
          else:
            yield u,j,i
                                               

class MixedSampler(Sampler):
    """ Mixed sampler between UniformPair() sampler, which samples cid 
    more if more checkins have been observed in context cid;
    and UniformUserUniformItem() sampler, which samples cid with equal probability
    The UniformPair() sampler can exploit harder comparisons when both
    venues are observed; while the UniformUserUniformItem() sampler can explore a bit
    to also account for contexts that are not fully observed.
    It turned out that a combination of both strategies gets the strength of both.
    In our project, we had a LOO CV to test the performance under each one of the
    context. Taking in UniformPair() will boost the AUC a lot 
    UniformUserUniformItem() (uniform_user_ratio=1) gets an AUC of 0.653
    MixedSampler() with uniform_user_ratio=0.4 gets an AUC of 0.682
    """
    def __init__(self,
                sample_negative_items_empirically,
                random_state=None, 
                uniform_user_ratio=0.4):
        super(self.__class__, self).__init__(sample_negative_items_empirically, random_state)
        self.uniform_user_ratio = uniform_user_ratio # how much we use UniformUser sampler

    def generate_user(self):
        if self.rds.rand() > self.uniform_user_ratio:
          idx = self.rds.randint(0, self.data.nnz)
          u = self.user_occurence[idx]  # UniformPair sampler
        else:
          u = self.uniform_user()       # UniformUserUniformItem sampler
        return u
        
    def generate_samples(self,max_samples=None):
        """ generate a list of tuples of (cid,i,j) from the training data
        @Parameters
        -----------------------------------
        | max_samples: int, size of the list
        """
        self.max_samples = max_samples
        self.user_occurence = self.data.nonzero()[0]
        for _ in xrange(self.num_samples(self.data.nnz)):
          u = self.generate_user()
          i = self.rds.choice(self.data[u].indices)
          while self.mask[u,i] == True:
            u = self.generate_user()
            i = self.rds.choice(self.data[u].indices)
            
          j = self.random_item()
          while self.mask[u,j] == True or self.data[u,i] == self.data[u,j]:
            j = self.random_item()
          if self.data[u,i] > self.data[u,j]:
            yield u,i,j
          else:
            yield u,j,i


if __name__ == '__main__':
  import doctest, pickle

  data = pickle.load(open('../recommender/data.pkl','r'))
  model = FM(K=20, random_state=1).init(data)
  model.sampler = MixedSampler(True, model.random_state, 0.4)
  
  # doctest class functions
  doctest.testmod(
      extraglobs = {'model': model, 'data':data},
      verbose = True
      )

  ## Run Program:
  ## import pickle
  ## data = pickle.load(open('../recommender/data.pkl','r')) 
  ## model=est.FM(K=20,train=False,random_state=1)
  ## model.fit(data, num_iters=220, mode='cv', n_folds=5, stop=1) # for evaluate
  ## model.fit(data, num_iters=220, mode='fit',n_folds=5, stop=1) # for train
