% Import files from Swadlow, save as hdf5

clear;

file1='2002_1BN1.xls'
file2='2002_1BN2.xls'
file3='2008_1E.xls'
file4='2008_2A1.xls'

importfile(file1)
data_02_1BN1 = data(:,16:-1:1)./500; 

importfile(file2)
data_02_1BN2 = data(:,16:-1:1)./500;
%2002 data supplied as mV times a gain of 500


importfile(file3)
data_08_1E = data(:,16:-1:1);

importfile(file4)
data_08_2A1 = data(:,16:-1:1);

clear data;

dv1 = mean(data_02_1BN1(1:41,:));
dv2 = mean(data_02_1BN2(1:41,:));
dv3 = mean(data_08_1E(1:41,:));
dv4 = mean(data_08_2A1(1:41,:));


data_02_1BN1_v2 = zeros(size(data_02_1BN1));
data_02_1BN2_v2 = zeros(size(data_02_1BN2));
data_08_1E_v2 = zeros(size(data_08_1E));
data_08_2A1_v2 = zeros(size(data_08_2A1));

for i = 1:16
    data_02_1BN1_v2(:,i) = data_02_1BN1(:,i) - dv1(i);
    data_02_1BN2_v2(:,i) = data_02_1BN2(:,i) - dv2(i);
    data_08_1E_v2(:,i) = data_08_1E(:,i) - dv3(i);
    data_08_2A1_v2(:,i) = data_08_2A1(:,i) - dv4(i);
end

% for i = 1:16
%     data_02_1BN1_v2(:,i) = data_02_1BN1(:,i) - data_02_1BN1(1,i);
%     data_02_1BN2_v2(:,i) = data_02_1BN2(:,i) - data_02_1BN2(1,i);
%     data_08_1E_v2(:,i) = data_08_1E(:,i) - data_08_1E(1,i);
%     data_08_2A1_v2(:,i) = data_08_2A1(:,i) - data_08_2A1(1,i);
% end

t = [0:239]*0.025-1;

hdf5write('Swadlow_LFP.h5',...
    'data_02_1BN1',data_02_1BN1,...
    'data_02_1BN2',data_02_1BN2,...
    'data_08_1E',data_08_1E,...
    'data_08_2A1',data_08_2A1,...
    't',t)

hdf5write('Swadlow_LFP_v2.h5',...
    'data_02_1BN1',data_02_1BN1_v2,...
    'data_02_1BN2',data_02_1BN2_v2,...
    'data_08_1E',data_08_1E_v2,...
    'data_08_2A1',data_08_2A1_v2,...
    't',t)

%transposing for CSDplotter
data_02_1BN1_t = data_02_1BN1(:,16:-1:1)';
data_02_1BN1_v2_t = data_02_1BN1_v2(:,16:-1:1)';
data_02_1BN2_t = data_02_1BN2(:,16:-1:1)';
data_02_1BN2_v2_t = data_02_1BN2_v2(:,16:-1:1)';
data_08_1E_t = data_08_1E(:,16:-1:1)';
data_08_1E_v2_t = data_08_1E_v2(:,16:-1:1)';
data_08_2A1_t = data_08_2A1(:,16:-1:1)';
data_08_2A1_v2_t = data_08_2A1_v2(:,16:-1:1)';

%save transposed data in matlab format
save swadlows.mat data*_t