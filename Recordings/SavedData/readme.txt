Notes on what to find in each csv file:
- [...]_interaction_time_large -> data ordered and merged by the interaction time. the recording time is removed from the data.

- [...]_individualTS_large -> data placed together but not merged by the interaction time or any other time. Each group has the data (MG or DG)along side its own timeline, either recording time or the interaction time

- [...]_interaction_time -> data ordered and merged by the interaction time. the recording time is removed from the data. the interaction time has limited to two decimal points (23.023 -> 23.02) and duplicates are removed. This way the files aren't too large, and the maximum values per second is 100 (00-99). 


When merged, the interaction_time_large files don't have removed the NA in case one interaction time is not existing in that group. For instance, if one group has a data for the interaction time 23.023s and another for 23.025, both of these are saved, and na is added for the missing one (na for the 23.023 for the second group and na for the 23.025 for the first one). This applies to the interaction_time files too.


