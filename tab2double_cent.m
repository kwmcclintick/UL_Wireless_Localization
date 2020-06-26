clc
clear

mat = dir('*.mat');
for q = 1:length(mat)
    new = mean(table2array(load(mat(q).name).file.veh_pos_data(:,3:4)),1);
    file = load(mat(q).name);
    file.veh_pos_data = new;
    save(mat(q).name,'file')
end



