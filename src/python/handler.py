from logging import Handler

class IrisHandler(Handler):

    def __init__(self,bo):
        Handler.__init__(self)
        self.bo = bo

    def emit(self,record):

        self.bo.log_info(str(record).split("\"")[1])