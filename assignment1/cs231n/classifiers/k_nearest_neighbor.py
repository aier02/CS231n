import numpy as np

class KNearestNeighbor(object):
  """ a kNN classifier with L2 distance """
  #l2距离也称为欧式距离，所有对应分量差值的平方和再开方
  def __init__(self):
    pass
  #指定训练集，test数据和train数据的尺寸都已经转化为[样本数目，一个样本的所有pixel（原图片矩阵全部拉伸为一行vector）]
  def train(self, X, y):
    """
    Train the classifier. For k-nearest neighbors this is just 
    memorizing the training data.

    Inputs:
    - X: A numpy array of shape (num_train, D) containing the training data
      consisting of num_train samples each of dimension D.一行表示一个样本
    - y: A numpy array of shape (N,) containing the training labels, where
         y[i] is the label for X[i].
    """
    self.X_train = X
    self.y_train = y
  #k=1为最近邻算法
  def predict(self, X, k=1, num_loops=0):
    """
    Predict labels for test data using this classifier.

    Inputs:
    - X: A numpy array of shape (num_test, D) containing test data consisting
         of num_test samples each of dimension D.
    - k: The number of nearest neighbors that vote for the predicted labels.
    - num_loops: Determines which implementation to use to compute distances
      between training points and testing points.

    Returns:
    - y: A numpy array of shape (num_test,) containing predicted labels for the
      test data, where y[i] is the predicted label for the test point X[i].  
    """
    if num_loops == 0:
      dists = self.compute_distances_no_loops(X)
    elif num_loops == 1:
      dists = self.compute_distances_one_loop(X)
    elif num_loops == 2:
      dists = self.compute_distances_two_loops(X)
    else:
      raise ValueError('Invalid value %d for num_loops' % num_loops)

    return self.predict_labels(dists, k=k)
  #用最简单的两层循环计算dists矩阵
  def compute_distances_two_loops(self, X):
    #计算测试样本和每一个训练样本的距离，存储需求高，效率低
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using a nested loop over both the training data and the 
    test data.

    Inputs:
    - X: A numpy array of shape (num_test, D) containing test data.

    Returns:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      is the Euclidean distance between the ith test point and the jth training
      point.
    """
    #传入的x为测试样本，训练样本已经初始化
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    #dists矩阵dist[i,j]存储了test[i]和train[j]的l2距离，即每个pixel值的差值平方和开方
    dists = np.zeros((num_test, num_train))
    #循环次数为num_test*num_train，存储量大
    for i in range(num_test):
      for j in range(num_train):
        #####################################################################
        # TODO:                                                             #
        # Compute the l2 distance between the ith test point and the jth    #
        # training point, and store the result in dists[i, j]. You should   #
        # not use a loop over dimension.                                    #
        #####################################################################
        #np中矩阵相减为对应元素相减，square为每个元素乘方，sum为所有元素之和
        dists[i,j]=np.sqrt(np.sum(np.square(X[i]-self.X_train[j])))
        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
    return dists
  #只用一层循环计算dists矩阵
  def compute_distances_one_loop(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using a single loop over the test data.

    Input / Output: Same as compute_distances_two_loops
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    for i in range(num_test):
      #######################################################################
      # TODO:                                                               #
      # Compute the l2 distance between the ith test point and all training #
      # points, and store the result in dists[i, :].                        #
      #######################################################################
      #每次计算为第i个测试样本和所有训练样本的l2距离,实质和双层循环没有区别,区别在于np.
      #sum在不指定axis时是所有元素相加得到一个实数，指定axis=n，表示固定第n维后求'所有
      #'元素之和，比如一个三维矩阵(2,row,col)，axis=0，则sum = a[0][n1][n2]+a[1][
      # n1][n2]
      #sum中axis=1意味着计算每一行元素之和，返回一行，每一个元素对应test[i]与train的
      #l2距离
      dists[i,:]=np.sqrt(np.sum(np.square(X[i]-self.X_train),axis=1))
      #######################################################################
      #                         END OF YOUR CODE                            #
      #######################################################################
    return dists

  def compute_distances_no_loops(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using no explicit loops.

    Input / Output: Same as compute_distances_two_loops
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train)) 
    #########################################################################
    # TODO:                                                                 #
    # Compute the l2 distance between all test points and all training      #
    # points without using any explicit loops, and store the result in      #
    # dists.                                                                #
    #                                                                       #
    # You should implement this function using only basic array operations; #
    # in particular you should not use functions from scipy.                #
    #                                                                       #
    # HINT: Try to formulate the l2 distance using matrix multiplication    #
    #       and two broadcast sums.                                         #
    #########################################################################
    #根据l2距离的定义，将每个（xi-xj）^2拆分为xi^2-2xi*xj+xj^2,xi*xj由矩阵的点积实现
    #将x-train转置，再和test数据进行点积
    dists=np.multiply(np.dot(X,self.X_train.T),-2)
    #test和train每行pixel值乘方后，即每一个样本的所有pixel值乘方，进行求和
    res1=np.sum(np.square(X),axis=1)
    res2=np.sum(np.square(self.X_train),axis=1)
    dists=np.sqrt(dists+res1+res2)
    #########################################################################
    #                         END OF YOUR CODE                              #
    #########################################################################
    return dists

  def predict_labels(self, dists, k=1):
    """
    Given a matrix of distances between test points and training points,
    predict a label for each test point.


    Inputs:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      gives the distance betwen the ith test point and the jth training point.

    Returns:
    - y: A numpy array of shape (num_test,) containing predicted labels for the
      test data, where y[i] is the predicted label for the test point X[i].  
    """
    num_test = dists.shape[0]
    y_pred = np.zeros(num_test)
    for i in range(num_test):
      # A list of length k storing the labels of the k nearest neighbors to
      # the ith test point.
      closest_y = []
      #########################################################################
      # TODO:                                                                 #
      # Use the distance matrix to find the k nearest neighbors of the ith    #
      # testing point, and use self.y_train to find the labels of these       #
      # neighbors. Store these labels in closest_y.                           #
      # Hint: Look up the function numpy.argsort.                             #
      #########################################################################
      #np.argsort(a,axis)返回数组a中数值从小到大的索引值，axis=0按列排序，axis=1按行排
      #序，选择前k个；这里求得dists矩阵第i行前k小的数据的索引，然后根据索引找到对应的y值
      #closest_y存储的是k个紧邻的训练样本种类
      closest_y=self.y_train[np.argsort(dists[i])[0:k]]
      #########################################################################
      # TODO:                                                                 #
      # Now that you have found the labels of the k nearest neighbors, you    #
      # need to find the most common label in the list closest_y of labels.   #
      # Store this label in y_pred[i]. Break ties by choosing the smaller     #
      # label.                                                                #
      #########################################################################
      #根据多数表决决定y_pred的值,np.bincount（b）返回数组a，a[i]表示数组b中值为i的个数
      #np.argmax()返回数组中最大的元素值，若出现相同的最大值，取位置靠前者；下面找出种类
      #出现次数最多者作为测试的预测种类
      y_pred[i]=np.argmax(np.bincount(closest_y))
      #########################################################################
      #                           END OF YOUR CODE                            # 
      #########################################################################

    return y_pred

