
def writeLogFile(file_path, message, write_mode='a'):
    """WriteLogFile
    -------------------------------------------------------------------------
    DESCRIPTION:

    Write a given message in a text file according to a given file path.
    -------------------------------------------------------------------------
    INPUTS:
    1. file_path: string that represent the path to the text file. Including
                  the name of the file and its extension.
                  --> Ex: '/home/myStudy/DATA/log/log_error.txt'
    2. message: String that represent the message to write in the text file.
    3. write_mode: The write that will define if we create a new file (replace
                   file) or if we append the message to an existing file (if
                   the file exist). (Default: 'a')
                   --> Options: - No argument
                                - 'a'
                                - 'w'
    -------------------------------------------------------------------------
    AUTHOR(S): MEDomicsLab consortium
    -------------------------------------------------------------------------
    STATEMENT:
    This file is part of <https://github.com/MEDomics/MEDomicsLab/>,
    a package providing MATLAB programming tools for radiomics analysis.
     --> Copyright (C) MEDomicsLab consortium.

    This package is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This package is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this package.  If not, see <http://www.gnu.org/licenses/>.
    -------------------------------------------------------------------------
    """
    file_object = open(file_path, write_mode)
    file_object.write(message)
    file_object.close()
