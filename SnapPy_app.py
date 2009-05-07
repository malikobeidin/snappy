import IPython
import Tkinter as Tk_
import os, sys
import re

DefaultFonts = {'darwin': ('Monaco', 18, 'normal'),
                'linux2': ('fixed', 18, 'normal')
                }
def default_font():
    try:
        return DefaultFonts[sys.platform]
    except:
        return 'Helvetica'

ansi_seqs = re.compile('(?:\x01*\x1b\[((?:[0-9]*;)*[0-9]*.)\x02*)*([^\x01\x1b]*)',
                       re.MULTILINE)

ansi_colors =  {'0;30m': 'Black',
                '0;31m': 'Red',
                '0;32m': 'Green',
                '0;33m': 'Brown',
                '0;34m': 'Blue',
                '0;35m': 'Purple',
                '0;36m': 'Cyan',
                '0;37m': 'LightGray',
                '1;30m': 'DarkGray',
                '1;31m': 'DarkRed',
                '1;32m': 'SeaGreen',
                '1;33m': 'Yellow',
                '1;34m': 'LightBlue',
                '1;35m': 'MediumPurple',
                '1;36m': 'LightCyan',
                '1;37m': 'White'}

class TkTerm:
    """
    A Tkinter terminal window that runs an IPython shell.
    Some ideas borrowed from code written by Eitan Isaacson, IBM Corp.
    """
    def __init__(self, the_shell, name='TkTerm', root=None):
        try:
            self.banner = the_shell.banner
        except:
            self.banner = the_shell.IP.BANNER
        self.window = window = Tk_.Tk(root)
        window.protocol("WM_DELETE_WINDOW", self.close)
        self.frame = frame = Tk_.Frame(window)
        self.text = text = Tk_.Text(frame,
                                    font=default_font(),
                                    background='#e00fffe00'
                                )
        self.scroller = scroller = Tk_.Scrollbar(frame, command=text.yview)
        text.config(yscrollcommand = scroller.set)
        scroller.pack(side=Tk_.RIGHT, fill=Tk_.Y, pady=10)
        text.pack(fill=Tk_.BOTH, expand=Tk_.YES)
        frame.pack(fill=Tk_.BOTH, expand=Tk_.YES)
        text.focus_set()
        text.bind('<KeyPress>', self.handle_keypress)
        text.bind('<Return>', self.handle_return)
        text.bind('<BackSpace>', self.handle_backspace)
        text.bind('<Delete>', self.handle_backspace)
        text.bind('<Up>', self.handle_up)
        text.bind('<Down>', self.handle_down)
        text.bind('<<Copy>>', self.copy)
        text.bind('<<Paste>>', self.paste)
        text.bind('<<Cut>>', lambda event : 'break')   # disabled
        text.bind('<<Clear>>', lambda event : 'break') # disabled
        # self.end_index marks the end of the text written by us.
        # Everything above this position should be
        # immutable, and tagged with the "output" style.
        self.end_index = self.text.index(Tk_.INSERT)
        # Mark immutable text with a different background.
        text.tag_config('output', background='White')
        # But don't override the cut-paste background.
        text.tag_lower('output') 
        # Style tags for colored text. 
        for code in ansi_colors:
            text.tag_config(code, foreground=ansi_colors[code])
        self.banner = the_shell.banner
        self.IP = the_shell.IP
        self.In = self.IP.user_ns['In']
        self.history_pointer=0
        self.saved_line=''
        self.IP.write = self.write                 # used for the prompt
        IPython.Shell.Term.cout.write = self.write # used for output
        IPython.Shell.Term.cerr.write = self.write # used for tracebacks
        sys.stdout = self # also used for tracebacks (why???)
        sys.displayhook = self.IP.outputcache
        self.copy_buffer=''
        self.start_interaction()

    def close(self):
        self.live = False
        self.window.update_idletasks()
        self.window.quit()

    def close_event(self, event):
        self.close()

    def handle_keypress(self, event):
        if event.char == '\003':
            raise KeyboardInterrupt
        if event.char == '\004':
            self.close()
            return
        if self.text.compare(Tk_.INSERT, '<', self.end_index):
            self.text.mark_set(Tk_.INSERT, self.end_index)

    def handle_return(self, event):
        line=self.text.get(self.end_index, Tk_.END)
        self.text.tag_add('output', self.end_index, Tk_.END)
        self.end_index = self.text.index(Tk_.END)
        self.send_line(line)
        return 'break'

    def handle_backspace(self, event):
        if self.text.compare(Tk_.INSERT, '<=', self.end_index):
            self.window.bell()
            return 'break'

    def handle_up(self, event):
        if self.history_pointer >= len(self.In):
            self.window.bell()
            return 'break'
        if self.history_pointer == 0:
            self.saved_line = self.text.get(self.end_index, Tk_.END)
        self.text.delete(self.end_index, Tk_.END)
        self.history_pointer += 1
        self.write(self.In[-self.history_pointer].strip('\n'),
                   style=(), mutable=True)
        self.text.mark_set(Tk_.INSERT, Tk_.END)
        return 'break'

    def handle_down(self, event):
        self.text.delete(self.end_index, Tk_.END)
        if self.history_pointer == 0:
            self.window.bell()
            return 'break'
        self.history_pointer -= 1
        if self.history_pointer == 0:
            self.write(self.saved_line.strip('\n'),
                       style=(), mutable=True)
        else:
            self.write(self.In[-self.history_pointer].strip('\n'),
                       style=(), mutable=True)
        self.text.mark_set(Tk_.INSERT, Tk_.END)
        return 'break'
    
    def copy(self, event):
        self.copy_buffer = self.text.get(Tk_.SEL_FIRST, Tk_.SEL_LAST)
        return 'break'

    def paste(self, event):
        if self.text.compare(Tk_.INSERT, '>=', self.end_index):
            self.text.insert(Tk_.INSERT, self.copy_buffer)
        return 'break'

    def start_interaction(self):
        """
        Print the banner and issue the first prompt.
        """
        self.text.tag_config('banner', foreground='DarkGreen')
        self.write(self.banner, style=('output', 'banner'))
        self.IP.interact_prompt()
        self.end_index = self.text.index(Tk_.INSERT)
 
    def send_line(self, line):
        """
        Send one line of input to the interpreter, who will write
        the result on our Text widget.  Then issue a new prompt.
        """
        self.write('\n')
        if self.IP.more:
            self.IP.rl_do_indent = True
        else:
            self.IP.rl_do_indent = False
        line = line.decode(self.IP.stdin_encoding)
        self.IP.interact_handle_input(line)
        self.IP.interact_prompt()
        self.text.see(Tk_.INSERT)
        self.end_index = self.text.index(Tk_.INSERT)
        self.text.insert(Tk_.INSERT, self.IP.indent_current_str(), ())
        self.text.delete(Tk_.INSERT, Tk_.END)
        self.history_pointer = 0

    def write1(self, string):
        self.write(string, style=('test1',))

    def write2(self, string):
        self.write(string, style=('test2',))
                   
    def write(self, string, style=('output',), mutable=False):
        """
        Writes a string containing ansi color escape sequences to our
        Text widget, starting at the end_index position.
        """
        self.text.mark_set(Tk_.INSERT, self.end_index)
        pairs = ansi_seqs.findall(string)
        for pair in pairs:
            code, text = pair
            tags = (code,) + style if code else style
            if text:
                self.text.insert(Tk_.INSERT, text, tags)
        if mutable is False:
            self.end_index = self.text.index(Tk_.INSERT)
        self.text.see(Tk_.INSERT)

    def flush(self):
        """
        Since we are pretending to be an IOTerm.
        """
        pass

if __name__ == "__main__":
    import SnapPy
    from SnapPy.SnapPy_shell import the_shell
    SnapPy_ns = dict([(x, getattr(SnapPy,x)) for x in SnapPy.__all__])
    the_shell.IP.user_ns.update(SnapPy_ns)
    os.environ['TERM'] = 'dumb'
    terminal = TkTerm(the_shell)
    terminal.window.mainloop()
