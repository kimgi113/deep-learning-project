dataset = table();
pattern = 'psd_movingAve'; %de_moving, de_LDS, psd_moving, psd_LDS
band = 4; % 1=Delta, 2=theta, 3=alpha, 4=beta, 5=gamma

%% Session1에 해당하는 파일 불러오기

session1dataset = table();
filelist = dir('D:\Bogle\OneDrive - dgist.ac.kr\바탕 화면\과제\6-2\Deep learning\201122 midterm\dataset11 SEEDIV\eeg_feature_smooth\1\*.mat');
load('session1_label.mat');

filelength = 15;
filelistString1 = zeros(1,filelength);
filelistString1 = string(filelistString1);
for i = 1:filelength
   
    filelistString1(i) = filelist(i).name;
    
end

for k = 1:filelength
    filelistStruct = load(filelistString1(k));
    persondata = who('-file', filelistString1(k));
    persondata = natsortfiles(persondata);
    persondata = string(persondata);


    TF = contains(persondata, pattern);
    selectPersondata = persondata(TF);

    for i = 1:length(selectPersondata)
        
        selectdata = [filelistStruct.(selectPersondata(i))];
        bandselectdata = selectdata(:,:,band);
        newdataset = table({bandselectdata}, session1_label(i),k,i);
        
        session1dataset = vertcat(session1dataset, newdataset);
    end

end


session1dataset.Properties.VariableNames{'Var1'} = 'eegdata';
session1dataset.Properties.VariableNames{'Var2'} = 'label';
session1dataset.Properties.VariableNames{'k'} = 'ID';
session1dataset.Properties.VariableNames{'i'} = 'Video';

%% Session2에 해당하는 파일 불러오기
session2dataset = table();
filelist = dir('D:\Bogle\OneDrive - dgist.ac.kr\바탕 화면\과제\6-2\Deep learning\201122 midterm\dataset11 SEEDIV\eeg_feature_smooth\2\*.mat');
load('session2_label.mat');

filelength = length(filelist) -1;
filelistString1 = zeros(1,filelength);
filelistString1 = string(filelistString1);
for i = 1:filelength
   
    filelistString1(i) = filelist(i).name;
    
end

for k = 1:filelength
    filelistStruct = load(filelistString1(k));
    persondata = who('-file', filelistString1(k));
    persondata = natsortfiles(persondata);
    persondata = string(persondata);


    TF = contains(persondata, pattern);
    selectPersondata = persondata(TF);

    for i = 1:length(selectPersondata)
        
        selectdata = [filelistStruct.(selectPersondata(i))];
        bandselectdata = selectdata(:,:,band);
        newdataset = table({bandselectdata}, session2_label(i),k,i+24);
        
        session2dataset = vertcat(session2dataset, newdataset);
    end

end


session2dataset.Properties.VariableNames{'Var1'} = 'eegdata';
session2dataset.Properties.VariableNames{'Var2'} = 'label';
session2dataset.Properties.VariableNames{'k'} = 'ID';
session2dataset.Properties.VariableNames{'Var4'} = 'Video';


%% Session3에 해당하는 파일 불러오기
session3dataset = table();
filelist = dir('D:\Bogle\OneDrive - dgist.ac.kr\바탕 화면\과제\6-2\Deep learning\201122 midterm\dataset11 SEEDIV\eeg_feature_smooth\3\*.mat');
load('session3_label.mat');

filelength = length(filelist) -1;
filelistString1 = zeros(1,filelength);
filelistString1 = string(filelistString1);
for i = 1:filelength
   
    filelistString1(i) = filelist(i).name;
    
end

for k = 1:filelength
    filelistStruct = load(filelistString1(k));
    persondata = who('-file', filelistString1(k));
    persondata = natsortfiles(persondata);
    persondata = string(persondata);


    TF = contains(persondata, pattern);
    selectPersondata = persondata(TF);

    for i = 1:length(selectPersondata)
        
        selectdata = [filelistStruct.(selectPersondata(i))];
        bandselectdata = selectdata(:,:,band);
        newdataset = table({bandselectdata}, session3_label(i),k,i+48);
        
        session3dataset = vertcat(session3dataset, newdataset);
    end

end
session3dataset.Properties.VariableNames{'Var1'} = 'eegdata';
session3dataset.Properties.VariableNames{'Var2'} = 'label';
session3dataset.Properties.VariableNames{'k'} = 'ID';
session3dataset.Properties.VariableNames{'Var4'} = 'Video';


%% Session dataset 합치기

dataset = vertcat(session1dataset, session2dataset, session3dataset);
save('psddataset_beta', 'dataset', '-v7.3')
    
    