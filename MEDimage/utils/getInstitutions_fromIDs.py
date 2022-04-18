def getInstitutions_fromIDs(patientIDs):
    # -------------------------------------------------------------------------
    # function [institution_catVector] = getInstitutions_fromIDs(patientIDs)
    # -------------------------------------------------------------------------
    # DESCRIPTION: 
    # This function extracts the institution strings from a cell of patient
    # IDs.
    # -------------------------------------------------------------------------
    # INPUTS:
    # 1. patientIDs: Full path to the where a given variable data CSV 
    #                       table is stored.
    #                --> Ex: {'Cervix-UCSF-005';'Cervix-CEM-010'}
    # -------------------------------------------------------------------------
    # OUTPUTS: 
    # 1. institution_catVector: Categorical vector, specifying the institution
    #                           of each patientID entry in "patientIDs".
    #                           --> Ex: {UCSF;CEM}
    # -------------------------------------------------------------------------
    # AUTHOR(S):                                                              
    # - MEDomics consortium                                                   
    # -------------------------------------------------------------------------
    # HISTORY:                                                                
    # - Creation: March 2019                                                   
    # -------------------------------------------------------------------------
    # STATEMENT:                                                                                                                          
    # MEDomicsLab: An open-source computation platform for multi-omics        
    # modeling in medicine.                                                    
    # --> Copyright (C) 2019  MEDomics consortium                             
    #                                                                            
    #   This program is free software: you can redistribute it and/or modify  
    #   it under the terms of the GNU General Public License as published by  
    #   the Free Software Foundation, either version 3 of the License, or     
    #   (at your option) any later version.                                   
    #                                                                         
    #   This program is distributed in the hope that it will be useful,       
    #   but WITHOUT ANY WARRANTY; without even the implied warranty of        
    #   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         
    #   GNU General Public License for more details.                          
    #                                                                         
    #   You should have received a copy of the GNU General Public License     
    #   along with this program. If not, see <http://www.gnu.org/licenses/> . 
    # *************************************************************************

    nID = len(patientIDs)
    return patientIDs.str.rsplit('-', expand=True)[1]
