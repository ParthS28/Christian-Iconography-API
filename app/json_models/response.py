from pydantic import BaseModel

class Response(BaseModel):
    birth_virgin: float
    marriage: float 
    annunciation: float
    birth_jesus: float
    adoration: float 
    coronation: float
    assumption: float
    death: float
    virgin_and_child: float