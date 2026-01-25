import sys
import utils.logger as logger

def error_message_detail(error, error_detail: sys):
    '''
    Extracts detailed information about an exception including the file name and line number.
    '''
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    error_msg = f"Error occurred in script: {file_name} at line number: {line_number} with error message: {str(error)}"
    return error_msg

class CustomException(Exception):
    '''
    Handler for custom exceptions that logs detailed error information.
    '''
    def __init__(self, error_message, error_detail: sys):
        '''
        Initializes the CustomException with a detailed error message.
        '''
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self):
        '''
        Returns the detailed error message as a string.
        '''
        return self.error_message