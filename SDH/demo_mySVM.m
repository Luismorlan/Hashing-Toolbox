function [B,tB] = demo_mySVM(traindata,testdata,traingnd,testgnd,codeLen,train)
% if sum(traingnd==0), 
%     traingnd = traingnd + 1;
%     testgnd = testgnd + 1;
% end
%  for noise = [0.001 0.02 ]
%     
% testdata_noise = imnoise(testdata,'gaussian',noise);
% traindata = imnoise(traindata, 'gaussian', 0.02);
% select training data

for Ntrain = train, 
            tic
            rand('seed',1);
            ix = randsample(size(traindata,1), Ntrain);
            X = traindata(ix,:); 
            if exist('traingnd','var')
                label = double(traingnd(ix,:));
            elseif exist('Y_train','var')
                label = double(traingnd(ix,:));
            end

% Ntrain = size(traindata,1);
% X = traindata;
% label = double(traingnd);




% get anchors
            n_anchors = 1000;
            % rand('seed',1);
            anchor = X(randsample(Ntrain, n_anchors),:);
            

            % % determin rbf width sigma
            % Dis = EuDist2(X,anchor,0);
            % % sigma = mean(mean(Dis)).^0.5;
            % sigma = mean(min(Dis,[],2).^0.5);
            % clear Dis 
            sigma =mean(mean(sqdist(X,anchor)))/2;

            PhiX = exp(-sqdist(X,anchor)/(2 *sigma));
            PhiX = [PhiX, ones(Ntrain,1)];
            % sigma =mean(mean(sqdist(testdata,anchor)))/2;
            Phi_testdata = exp(-sqdist(testdata,anchor)/(2  * sigma)); 
            Phi_testdata = [Phi_testdata, ones(size(Phi_testdata,1),1)];
            % sigma =mean(mean(sqdist(traindata,anchor)))/2;
            Phi_traindata = exp(-sqdist(traindata,anchor)/(2  *sigma)); 
            Phi_traindata = [Phi_traindata, ones(size(Phi_traindata,1),1)];

            % learn G and F
            maxItr = 3;
            gmap.lambda = 1; gmap.loss = 'L2';
            Fmap.type = 'RBF'; Fmap.lambda = 1e-2; Fmap.sigmoid = false;
            Fmap.nu = 1e-5; %  penalty parm for F, 1e-5 for mySVM_binary, 1e-1 for mySVM;



            % trueTrainTest = trueTrainTest_set.trueTrainTest_500NN;

           %%
                nbits = codeLen; % low dimension L in the method

                % Init Z
                randn('seed',3);
                Zinit=sign(randn(Ntrain,nbits));


                debug = 0;
              
                            randn('seed',3);
                            Zinit=sign(randn(Ntrain,nbits));
                            debug = 0;
                            
                            addpath code\;
                            S = constructW(PhiX);
                            L = diag(sum(S))-S;
                            L1 = ones(size(L));
                            [R,~,D]=svd(label'*L1*label);
                            label = label*R;
                            
                            [~, F, ~] = mySVM_binary(PhiX,label,Zinit,gmap,Fmap,[],maxItr,debug);
            %     [~, F, ~] = mySVM_binary_stochastic(PhiX,label,Zinit,gmap,Fmap,[],maxItr,debug);
%                            F_cifar_5000 = [F_cifar_5000; F ];
            
                % continuous, no binary constraint
                % Zinit=randn(Ntrain,nbits);
                % [G, F, ZX] = mySVM(PhiX,label,Zinit,gmap,Fmap,[],maxItr);


                % ------------------------ Method 1 ------------------------------------


                            FX = [Phi_traindata ;Phi_testdata(1:5000,:)]*F.W; 
                            tFX = Phi_testdata(5001:end,:)*F.W;


                            B = FX > 0;
                            tB = tFX > 0;

    end
end




