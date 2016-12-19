more off
alpha=.01
num_iter=15
batch_size=1

% load and normalize data
data = load('~/gd_algorithms/data/test_multi.txt');
m = rows(data);
n = columns(data) - 1;
mu = mean(data);
sigma = std(data);
data_norm = (data .- mu) ./ sigma;
Xy = [ones(m,1) data_norm(:,1:end-1) data(:,end)];
batches = ceil(m / batch_size);
theta = zeros(n+1,1);
cost_hist = Inf(num_iter*batches,1);

for i = 1:num_iter
  Xy_rand = Xy(randperm(end),:);
  for b = 1:batches
    B = Xy(((b-1)*batch_size + 1):min(b*batch_size,m),:);
    B_x = B(:,1:end-1);
    B_y = B(:,end);
    loss_B = B_x * theta - B_y ;
    grad = B_x' * loss_B / rows(B_x);
    theta = theta - alpha * grad;
    cost_hist((i-1)*batches+b,1) = sum((Xy(:,1:end-1) * theta - Xy(:,end)).^2) / (2 * m);
   endfor
endfor

theta
x = [1650 3]
x_norm = (x - mu(:,1:end-1)) ./ sigma(:,1:end-1);
predict = theta' * [1 x_norm]'
plot(cost_hist, 'linewidth', 1)




