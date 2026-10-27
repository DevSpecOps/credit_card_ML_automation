import pandas as pd
import pandera.pandas as pa
from pandera.typing import Series


class RawCreditSchema(pa.DataFrameModel):
    ID: Series[int] = pa.Field(ge=1)
    LIMIT_BAL: Series[float] = pa.Field(gt=0, le=1_000_000)
    SEX: Series[int] = pa.Field(isin=[1, 2])
    EDUCATION: Series[int] = pa.Field(isin=[0, 1, 2, 3, 4, 5, 6])
    MARRIAGE: Series[int] = pa.Field(isin=[0, 1, 2, 3])
    AGE: Series[int] = pa.Field(ge=18, le=100)
    PAY_0: Series[int] = pa.Field(ge=-2, le=9)
    PAY_2: Series[int] = pa.Field(ge=-2, le=9)
    PAY_3: Series[int] = pa.Field(ge=-2, le=9)
    PAY_4: Series[int] = pa.Field(ge=-2, le=9)
    PAY_5: Series[int] = pa.Field(ge=-2, le=9)
    PAY_6: Series[int] = pa.Field(ge=-2, le=9)
    BILL_AMT1: Series[float] = pa.Field()
    BILL_AMT2: Series[float] = pa.Field()
    BILL_AMT3: Series[float] = pa.Field()
    BILL_AMT4: Series[float] = pa.Field()
    BILL_AMT5: Series[float] = pa.Field()
    BILL_AMT6: Series[float] = pa.Field()
    PAY_AMT1: Series[float] = pa.Field(ge=0)
    PAY_AMT2: Series[float] = pa.Field(ge=0)
    PAY_AMT3: Series[float] = pa.Field(ge=0)
    PAY_AMT4: Series[float] = pa.Field(ge=0)
    PAY_AMT5: Series[float] = pa.Field(ge=0)
    PAY_AMT6: Series[float] = pa.Field(ge=0)
    default_payment_next_month: Series[int] = pa.Field(isin=[0, 1])

    class Config:
        strict = False
        coerce = True


def validate_raw(df: pd.DataFrame) -> pd.DataFrame:
    return RawCreditSchema.validate(df, lazy=True)
