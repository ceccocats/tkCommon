if __name__ == "__main__":
    import glob, os
    os.chdir("/mydir")
    for file in glob.glob("*.h"):
        print(file)