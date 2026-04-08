from fastapi import FastAPI
from aion.monitor.hardware import get_cpu, get_ram, get_disk
from aion.monitor.runtime import tracker

app = FastAPI()

@app.get("/hardware")
def hardware():
    return {
        "cpu": get_cpu(),
        "ram": get_ram(),
        "disk": get_disk()
    }

@app.get("/runtime")
def runtime():
    return tracker.stats()