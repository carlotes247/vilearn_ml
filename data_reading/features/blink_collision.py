import datetime
import pandas as pd

class BlinkCollision:
    """
    Used to store information about a single blink collision
    """
    ref_TS: datetime
    adv_TS: datetime
    delta_ms: float

    def __init__(self, ref_TS: datetime, adv_TS: datetime, delta: float):
        self.ref_TS = ref_TS
        self.adv_TS = adv_TS
        self.delta_ms = delta    

class BlinkCollisionsList:
    """
    Stores info about all the blink collisions betweent two participants
    """
    ref_name: str
    adv_name: str    
    collisions: list[BlinkCollision]

    def __init__(self, ref_name: str, adv_name: str,):
        self.ref_name = ref_name
        self.adv_name = adv_name
        self.collisions = list()

    def add_collision(self, ref_TS: datetime, adv_TS: datetime, delta: float):
        new_collision = BlinkCollision(ref_TS, adv_TS, delta)
        self.collisions.append(new_collision)

    def get_collisions_df(self) -> pd.DataFrame:
        colls_df = pd.DataFrame([vars(x) for x in self.collisions])
        return colls_df
    
    def get_collisions_with_all_TS(self, timestamps: list[datetime.datetime]):
        """
        Pass in a list of timestamps and generates a complete dataframe where only the blink collisions are True
        """
        if (timestamps is None):
            raise Exception("Error, timestamp list is not initialised")
        if (len(timestamps) < len(self.collisions)):
            raise Exception("Can't create a dataframe smaller than the list of blink collisions!")
        TS_set = set(timestamps) # quicker lookups, removes repeated (also somehow "ts in list" returns false, but "ts in set" returns true. No idea why)
        length_df = len(TS_set)
        ref_dict = dict(zip(timestamps, [False] * length_df))
        adv_dict = dict(zip(timestamps, [False] * length_df))
        for collision in self.collisions:
            if (collision.ref_TS not in TS_set or collision.adv_TS not in TS_set):
                raise Exception("The blink collisions are not in timestamps")
            ref_dict[collision.ref_TS] = True
            adv_dict[collision.adv_TS] = True
        df_all_TS = pd.DataFrame({f'blink_onsets_{self.ref_name}_ref': ref_dict.values(),
                f'blink_onsets_{self.adv_name}_adv': adv_dict.values(),
                }, index=ref_dict.keys())
        return df_all_TS.astype(int)

        