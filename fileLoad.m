dataset = table();

%% Session1�� �ش��ϴ� ���� �ҷ�����

session1dataset = table();
filelist = dir('D:\Bogle\OneDrive - dgist.ac.kr\���� ȭ��\����\6-2\Deep learning\201122 midterm\dataset11 SEEDIV\eeg_raw_data\1\*.mat');
load('session1_label.mat')

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

% for i = 1:5
%     plot(filelistStruct.(persondata(1))(i,:));
%     hold on
% end

    for i = 1:length(persondata)
    
        eegdata = [filelistStruct.(persondata(i))];
        newdataset = table({eegdata}, session1_label(i),k,i);

        session1dataset = vertcat(session1dataset, newdataset);
    end 
end


session1dataset.Properties.VariableNames{'Var1'} = 'eegdata';
session1dataset.Properties.VariableNames{'Var2'} = 'label';
session1dataset.Properties.VariableNames{'k'} = 'ID';
session1dataset.Properties.VariableNames{'i'} = 'Video';


%% Session2�� �ش��ϴ� ���� �ҷ�����

session2dataset = table();
filelist = dir('D:\Bogle\OneDrive - dgist.ac.kr\���� ȭ��\����\6-2\Deep learning\201122 midterm\dataset11 SEEDIV\eeg_raw_data\2\*.mat');
load('session2_label.mat')

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

% for i = 1:5
%     plot(filelistStruct.(persondata(1))(i,:));
%     hold on
% end

    for i = 1:length(persondata)
    
        eegdata = [filelistStruct.(persondata(i))];
        newdataset = table({eegdata}, session2_label(i),k,i+24);

        session2dataset = vertcat(session2dataset, newdataset);
    end 
end


session2dataset.Properties.VariableNames{'Var1'} = 'eegdata';
session2dataset.Properties.VariableNames{'Var2'} = 'label';
session2dataset.Properties.VariableNames{'k'} = 'ID';
session2dataset.Properties.VariableNames{'Var4'} = 'Video';


%% Session3�� �ش��ϴ� ���� �ҷ�����

session3dataset = table();
filelist = dir('D:\Bogle\OneDrive - dgist.ac.kr\���� ȭ��\����\6-2\Deep learning\201122 midterm\dataset11 SEEDIV\eeg_raw_data\3\*.mat');
load('session3_label.mat')

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

% for i = 1:5
%     plot(filelistStruct.(persondata(1))(i,:));
%     hold on
% end

    for i = 1:length(persondata)
    
        eegdata = [filelistStruct.(persondata(i))];
        newdataset = table({eegdata}, session3_label(i),k,i+48);

        session3dataset = vertcat(session3dataset, newdataset);
    end 
end


session3dataset.Properties.VariableNames{'Var1'} = 'eegdata';
session3dataset.Properties.VariableNames{'Var2'} = 'label';
session3dataset.Properties.VariableNames{'k'} = 'ID';
session3dataset.Properties.VariableNames{'Var4'} = 'Video';


%% Session dataset ��ġ��

dataset = vertcat(session1dataset, session2dataset, session3dataset);
save('data_ID', 'dataset', '-v7.3')
    
    