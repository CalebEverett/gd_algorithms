NB. This function takes an m x n matrix of numerical values
NB. with training examples (X) in columns 1 through n-1 and
NB. outcomes in column n as its argument and returns
NB. a 1 x n+1 vector of model parameters.

    gd_normal_eq =: 3 : 0
X =: (#y) $ 1 ,. (<a:;<<_1) { y
dot =: (+/ . *)
theta =: (%.(|:X) dot X) dot (|:X) dot (<a:;_1) { y
)


NB. Load test data and run test

load 'tables/csv'
data =: ". > readcsv '/root/gd_algorithms/j/test_multi.txt'
$ data

test=: 3 : 0
  assert. (<. gd_normal_eq data) -: 89597 139 _8738
  'test_gd_norm_eq passed'
)

smoutput test''

