import psutil

def get_cpu():
    return psutil.cpu_percent(interval=0.5)

def get_ram():
    mem = psutil.virtual_memory()
    return {
        "total": mem.total,
        "used": mem.used,
        "percent": mem.percent
    }

def get_disk():
    disk = psutil.disk_usage('/')
    return {
        "total": disk.total,
        "used": disk.used,
        "percent": disk.percent
    }