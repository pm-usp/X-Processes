import tm

def __main__():
    print("Hello, world!")

    tm.measure_time(print)
    tm.wrapper()
    #chama o arquivo de Excel
    tm.write_time_measurement('print',100)

__main__()