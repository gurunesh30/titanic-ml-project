from typing import Optional
from pydantic import BaseModel

class Passenger(BaseModel):
    pclass: int
    sex: str
    # Fixed: Changed parentheses () to square brackets []
    age: Optional[float] = None
    fare: Optional[float] = None