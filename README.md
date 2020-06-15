# PLS (Phone Lattice Scanning)

This technique focuses on searching the phone sequence of a keyword from the probability output of the LSTM model of a speech utterance.

This technique is divided into two main components:
1) **Formation of phone lattice**  
    A phone lattice is a graph where each node represents the phone detected by the LSTM model. The phones detected in a single time frame (1 time frame = 25ms in a utterance, difference between 2 time frames = 10ms) are aligned in the same column. Each phone/node in a 
    time frame is connected to every other phone/node in the consecutive time frame. These connections suggest the possible paths for 
    detection of the phone sequence. 

2) **Scanning the phone lattice for target phone sequence**  
    The scanning of the formed lattice is done using dynamic programming algorithm. There are three variations of dynamic programming 
    algorithm implemented for searching the targeted phone sequence. The input variables and their data type are commented in the code.
    The output for each phone lattice scanning method is explained as follows:  
    
    1) **Fixed PLS Single Pronunciation:**  
        Map of all detected sequences corresponding to the targeted keyword.  
        **Key:** Detected phone sequence  
        **Values:** Occurence probability of the detected phone sequence, start time frame of the detected sequence, pc value  
        
        **NOTE:** pc value is variable used during the algorithm and its value conventions do not hold any observatory significance.
        
    2) **Fixed PLS Multi Pronunciation:**  
        Map of all detected sequences corresponding to the targeted keyword.  
        **Key:** Detected phone sequence  
        **Values:** Occurence probability of the detected phone sequence, start time frame of the detected sequence, pc value array
        
    3) **Modified Dynamic PLS:**  
        Map of all detected sequences corresponding to the targeted keyword.  
        **Key:** Detected phone sequence  
        **Values:** Occurence probability of the detected phone sequence,  
                Edit distance of the detected phone sequence from the targeted phone sequence,  
                Cost of the detected phone sequence (this can be computed using the cost_func function from utils.py),  
                Start time frame,  
                End time frame of the detected sequence  
                
## Code Distribution

- probs/ folder contains the output sample of LSTM for code testing.
    - IDS Probabilities/ folder contains the insertion, deletion and substitution probabilities derived from LSTM model
    - LSTM Probabilities/ contains the probabilities of 1344 sentences of TIMIT TEST dataset from LSTM model
    
NOTE: Here the sample output for the TIMIT TEST dataset provided considers in total 39 phones which are mentioned in phone_mapping.json 
file. The phone folding from TIMIT standard phones to 39 phones is mentioned in folding.json file. 

- results/ folder contains the sample results obtained by considering keyword DARKSUIT and SUIT over 1344 sentences of TIMIT TEST dataset.

- process_lattice.py file forms the phone lattice from the LSTM output probabilities

- phone_lattice_scanning.py file contains all the phone scanning methods implemented.

Note that the codes are not restricted to the specific 39 phones structure. There is a blank ID phone provided in the code which depicts 
silence phone in the utterance. 

- testing.py file consists of two functions, test_one_utterance and test_multiple_utterance. These two functions can be used for testing
one or multiple utterances for one targeted phone sequence respectively.

## Addition information for running the code

- hspike and hnode are the parameters that are used to form phone lattice. The values of hspike and hnode should be in the range of 
[0.1, 0.5] and [5e-8, 5e-1] respectively. 

- The phone lattice algorithm requires phone number sequence as input. This input is the list of numbers mapping to phone as suggested
in phone_mapping.json file. In order to use the code, the user needs to provide the phone list of the targeted keyword. For instance, the 
the phone sequence of **DARKSUIT** is ["pau", "d", "aa", "r", "pau", "k", "s", "uw", "dx"]. In order to find the mapping of the phone to 
number mapping, use search_phone() function from utils.py file. 

- A demo file main.py is provided for reference. Place this main.py file outside the PLS folder and run the script.

- An extra file for testing is provided. keywords.txt and keywords.json consists of 80 keywords used in a research work. keywords.txt contains the list of these 80 keywords, and keywords.json consists the location and pronunciations of the corresponding keyword.<br/>
The format explanation of the keywords.json is as follows:<br/>
**Key:** String format keyword<br/>
**Values:** List of the files where the keyword is noticed,<br/>
        next is the list of the numbers which suggests whether the corresponding index file is present in TIMIT TEST dataset,<br/>
        phone sequence of the corresponding keyword of the correct American English pronunciation,<br/>
        phone to ID/number mapping of the correct pronunciation of phone sequence,<br/>
        phone number sequence of the pronuciation as noticed in the TIMIT TEST dataset .PHN file

