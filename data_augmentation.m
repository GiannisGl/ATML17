function done = data_augmentation(dname)
        foldername = fullfile(dname, 'classes/jpg/');
        folders = dir(foldername);
        disp('folder')
        disp(dname)
        %folders = dir('classes/masks/*'); %Uncomment for masks
        folders=folders(3:end,:);
        for folder=folders'
            path=strcat(foldername,folder.name);
            disp('path')
            disp(path)
            %path=strcat('classes/masks/',folder.name); %Uncomment for masks
            if ~isempty(strfind(folder.name, '_'))
                continue
            end
            files=dir(strcat(path,'/*.jpg'));
            %files=dir(strcat(path,'/*.png')); %Uncomment for masks
	   if ~exist(strcat(path,'h_fl'))
            mkdir(strcat(path,'_h_fl'));
            mkdir(strcat(path,'_v_fl'));
            mkdir(strcat(path,'_hv_fl'));
            mkdir(strcat(path,'_r_90'));
            mkdir(strcat(path,'_r_180'));
            mkdir(strcat(path,'_r_270'));
            mkdir(strcat(path,'_rnoise'));
	   end
            files=files(3:end,:);
            for file=files'
                disp('file')
                disp(file.name)
                I=imread(strcat(path,'/',file.name));
                I2 = flip(I ,2);           % horizontal flip
                I3 = flip(I ,1);           % vertical flip   
                I4 = flip(I3 ,2);          % horizontal and vertical flip
                
                R=imrotate(I, 90);            % Rotate 90
                R2=imrotate(I, 180);          % Rotate 180
                R3=imrotate(I, 180);          % Rotate 270
                
                n = round(rand(1)*3);

                switch n
                    case 0
                      mean=0;
                      var=rand(1)*0.5;
                      N=imnoise(I,'gaussian',mean,var);
                    case 1
                      N=imnoise(I,'poisson');
                    case 2
                      d = rand(1)*0.5;
                      N=imnoise(I,'salt & pepper',d);
                    case 3
                      v = rand(1);  
                      N = imnoise(I,'speckle',v); 
                  
                end
                
                imwrite(I2,strcat(path,'_h_fl/',file.name));
                imwrite(I3,strcat(path,'_v_fl/',file.name));
                imwrite(I4,strcat(path,'_hv_fl/',file.name));
                
                imwrite(R,strcat(path,'_r_90/',file.name));
                imwrite(R2,strcat(path,'_r_180/',file.name));
                imwrite(R3,strcat(path,'_r_270/',file.name));
                
                imwrite(N,strcat(path,'_rnoise/',file.name));
            %     figure(1);
            %     imshow(I);
            %     figure(2);
            %     imshow(N);
                
            end
            
        end
	done = 1;
end  
  
