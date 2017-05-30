function done = create_sub_dataset(dname, destfolder, txtfile)
      fname=fullfile(dname,txtfile);
      fid = fopen(fname);
      C = textscan(fid,'%s');
      fclose(fid);
      paths = C{1,1};
      disp(destfolder);
      [m,~] = size(paths)
      if ~exist(destfolder,'dir')
          mkdir(destfolder);
      end
      for i=1:m
          filename = paths{i,1};
          [pathstr,name,ext] = fileparts(filename) ;
          if ~exist(fullfile(destfolder,pathstr),'dir')
              mkdir(fullfile(destfolder,pathstr));
          end
          imgtmp = imread(fullfile(dname,filename));
          imwrite(imgtmp, fullfile(destfolder,filename));
       end
       
       done = 1;
end
