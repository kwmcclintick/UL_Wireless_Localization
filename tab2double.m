clc
clear

mat = dir('*.mat');
for q = 1:length(mat)
    new = table2array(load(mat(q).name).wheel_pos_data(:,3:4));
    file = load(mat(q).name);
    file.wheel_pos_data = new;
    save(mat(q).name,'file')
end



