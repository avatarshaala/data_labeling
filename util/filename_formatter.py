def dotjava_to_dotjson(java_fname, user, date):
    filename = (".").join(java_fname.split(".")[:-1])
    return f"{user}.{date}.{filename}.json"