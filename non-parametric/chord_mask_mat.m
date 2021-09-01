function M = chord_mask_mat(N, b, symmetric, self, random_order)
if ~exist('symmetric', 'var') || isempty(symmetric)
    symmetric = false;
end

if ~exist('self', 'var') || isempty(self)
    self = true;
end
if ~exist('random_order', 'var') || isempty(random_order)
    random_order = false;
end

if random_order
    rng(0, 'twister');
    ind = randperm(N);
end

M = zeros(N);
L = floor(log2(N) / log2(b));
for i=1:N
    if random_order
        M(i,ind(1+mod((i-1)+b.^(0:L-1), N))) = 1;
    else
        M(i,1+mod((i-1)+b.^(0:L-1), N)) = 1;
    end
    
    if symmetric
        if random_order
            M(i,ind(1+mod((i-1)-b.^(0:L-1), N))) = 1;
        else
            M(i,1+mod((i-1)-b.^(0:L-1), N)) = 1;
        end
    end
    
    if self
        M(i,i) = 1;
    end
end