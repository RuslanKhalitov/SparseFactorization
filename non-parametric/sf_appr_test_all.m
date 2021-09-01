datalist = readmatrix('datalist.csv', 'OutputType', 'char');
filenames = datalist(:,1);
categories = datalist(:,2);
nd = length(filenames);

mkdir results
max_iter = 1e5;

for di=1:nd
    fprintf('============= di=%d/%d =====================\n', di, nd);
    A = load_square_matrix(categories{di}, filenames{di});
    res = sf_appr_test(A, max_iter);
    fres = sprintf('results/sf_appr_test_di%d.mat', di);
    save(fres, 'res', 'filenames', 'categories', 'max_iter');
end




