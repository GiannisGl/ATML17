## Copyright (C) 2017 giannis
## 
## This program is free software; you can redistribute it and/or modify it
## under the terms of the GNU General Public License as published by
## the Free Software Foundation; either version 3 of the License, or
## (at your option) any later version.
## 
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
## 
## You should have received a copy of the GNU General Public License
## along with this program.  If not, see <http://www.gnu.org/licenses/>.

## -*- texinfo -*- 
## @deftypefn {Function File} {@var{retval} =} create_sub_dataset (@var{input1}, @var{input2})
##
## @seealso{}
## @end deftypefn

## Author: giannis <giannis@giannis-Lenovo>
## Created: 2017-05-26


function done = create_sub_dataset(dname, destfolder, txtfile)
      fname=fullfile(dname,txtfile);
      fid = fopen(fname);
      C = textscan(fid,'%s');
      fclose(fid);
      paths = C{1,1};
      disp(txtfile);
      [m,~] = size(paths)
      paths{2,1}
      mkdir(destfolder);
      if !exist(destfolder,'dir')
          mkdir(fullfile(destfolder,pathstr));
      end
      for i=1:1
          filename = paths{i,1};
          [pathstr,name,ext] = fileparts(filename) ;
          if !exist(pathstr,'dir')
              mkdir(fullfile(destfolder,pathstr));
          end
          imgtmp = imread(fullfile(dname,filename));
          imwrite(imgtmp, fullfile(destfolder,filename));
       end
       
       done = true;
end
