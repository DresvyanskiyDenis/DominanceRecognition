## Copyright (C) 2024 Dresvyanskiy
##
## This program is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with this program.  If not, see <https://www.gnu.org/licenses/>.

## -*- texinfo -*-
## @deftypefn {} {@var{retval} =} csvwrite_with_headers (@var{input1}, @var{input2})
##
## @seealso{}
## @end deftypefn

## Author: Dresvyanskiy <Dresvyanskiy@HIPPO>
## Created: 2024-02-15

function retval = csvwrite_with_headers (filename, data, headers)
    % Write headers to the file
    fid = fopen(filename, 'w');
    fprintf(fid, '%s,', headers{1:end-1});
    fprintf(fid, '%s\n', headers{end});
    fclose(fid);

    % Append data to the file
    dlmwrite(filename, data, '-append', 'delimiter', ',');
endfunction
