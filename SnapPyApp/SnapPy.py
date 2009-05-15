import IPython
import Tkinter as Tk_
import os, sys, re
from plink import LinkEditor
from tkFont import Font

DefaultFonts = {'darwin': ('Monaco', 16, 'normal'),
                'linux2': ('fixed', 16, 'normal')
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
                '1;30m': 'Black', #'DarkGray',
                '1;31m': 'DarkRed',
                '1;32m': 'SeaGreen',
                '1;33m': 'Yellow',
                '1;34m': 'LightBlue',
                '1;35m': 'MediumPurple',
                '1;36m': 'LightCyan',
                '1;37m': 'White'}

delims = re.compile(r'[\s\[\]\{\}\(\)\+\-\=\'`~!@#\$\^\&\*]+')

class TkTerm:
    """
    A Tkinter terminal window that runs an IPython shell.
    Some ideas borrowed from code written by Eitan Isaacson, IBM Corp.
    """
    def __init__(self, the_shell, name='TkTerm', root=None):
        self.window = window = Tk_.Tk(root)
        if (sys.platform == 'darwin') and hasattr(sys, 'frozen'):
            window.tk.call('console', 'hide')
        window.title(name)
        window.protocol("WM_DELETE_WINDOW", self.close)
        self.frame = frame = Tk_.Frame(window)
        self.text = text = Tk_.Text(frame,
                                    foreground='Black',
                                    borderwidth=3,
                                    background='#ec0fffec0',
                                    highlightthickness=0,
                                    relief=Tk_.FLAT
                                )
        self.set_font(default_font())
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
        text.bind('<Tab>', self.handle_tab)
        text.bind('<Up>', self.handle_up)
        text.bind('<Down>', self.handle_down)
        text.bind('<<Cut>>', self.protect_text)
        text.bind('<<Paste>>', self.paste)
        text.bind('<<Clear>>', self.protect_text)
        text.bind_all('<ButtonPress-2>', self.middle_mouse_down)
        text.bind_all('<ButtonRelease-2>', self.middle_mouse_up)
        text.bind('<Button-3>', lambda event : 'break')
        text.bind('<Button-4>', lambda event : 'break')
        text.bind('<MouseWheel>', lambda event : 'break')
        self.add_bindings()
        # 'output_end' marks the end of the text written by us.
        # Everything above this position should be
        # immutable, and tagged with the "output" style.
        self.text.mark_set('output_end', Tk_.INSERT)
        self.text.mark_gravity('output_end', Tk_.LEFT)
        text.tag_config('output')
        # Make sure we don't override the cut-paste background.
        text.tag_lower('output') 
        # Remember where we were when tab was pressed.
        self.tab_index = None
        self.tab_count = 0
        # Manage history
        self.hist_pointer=0
        self.hist_stem =''
        self.filtered_hist = []
        # Remember illegal pastes
        self.nasty = None
        self.nasty_text = None
        # Build style tags for colored text, 
        for code in ansi_colors:
            text.tag_config(code, foreground=ansi_colors[code])
        # and a style tag for messages.
        text.tag_config('msg', foreground='Red')
        self.build_menus()
        self.output_count = 0
        self.IP = the_shell.IP
        self.IP.magic_colors('LightBG')
        self.IP.write = self.write                 # used for the prompt
        IPython.Shell.Term.cout.write = self.write # used for output
        IPython.Shell.Term.cerr.write = self.write # used for tracebacks
        sys.stdout = self # also used for tracebacks (why???)
        sys.displayhook = self.IP.outputcache
        if the_shell.banner:
            self.banner = the_shell.banner
        else:
            cprt = 'Type "copyright", "credits" or "license" for more information.'
            self.banner = "Python %s on %s\n%s\n(%s)\n" %(
                sys.version, sys.platform, cprt,
                self.__class__.__name__)
        self.start_interaction()

    # For subclasses to override:
    def build_menus(self):
        pass

    # For subclasses to override:
    def add_bindings(self):
        pass

    def set_font(self, fontdesc):
        self.text.config(font=fontdesc)
        self.char_size = Font(self.text, fontdesc).measure('M')

    def close(self):
        self.live = False
        self.window.update_idletasks()
        self.window.quit()

    def close_event(self, event):
        self.close()

    def handle_keypress(self, event):
        self.clear_completions()
        if event.char == '\003':
            raise KeyboardInterrupt
        if event.char == '\004':
            self.close()
            return
        if self.text.compare(Tk_.INSERT, '<', 'output_end'):
            self.text.mark_set(Tk_.INSERT, 'output_end')

    def handle_return(self, event):
        self.clear_completions()
        line=self.text.get('output_end', Tk_.END)
        self.text.tag_add('output', 'output_end', Tk_.END)
        self.text.mark_set('output_end', Tk_.END)
        self.send_line(line)
        return 'break'

    def handle_backspace(self, event):
        self.clear_completions()
        if self.text.compare(Tk_.INSERT, '<=', 'output_end'):
            self.window.bell()
            return 'break'

    def handle_tab(self, event):
        self.tab_index = self.text.index(Tk_.INSERT)
        self.tab_count += 1
        if self.tab_count > 2:
            self.clear_completions()
            return 'break'
        line = self.text.get('output_end', self.tab_index).strip('\n')
        word = delims.split(line)[-1]
        completions = self.IP.complete(word)
        if len(completions) == 0:
            self.window.bell()
            return 'break'
        stem = self.stem(completions)
        if len(stem) > len(word):
            self.do_completion(word, stem)
        elif len(completions) > 25 and self.tab_count == 1:
            self.show_completions(['%s possibilities'%len(completions)])
        else:
            self.show_completions(completions)
        return 'break'

    def do_completion(self, word, completion):
        tail = completion[len(word):]
        self.text.insert(self.tab_index, tail)
        self.text.delete(Tk_.INSERT, Tk_.END)
        self.tab_index = None

    def show_completions(self, comps):
        width = self.text.winfo_width()
        font = Font(self.text, self.text.cget('font'))
        charwidth = width/self.char_size
        biggest = 2 + max([len(x) for x in comps])
        num_cols = charwidth/biggest
        num_rows = (len(comps) + num_cols -1)/num_cols
        rows = []
        format = '%%-%ds'%biggest
        for n in range(0, num_rows):
            rows.append(''.join([format%x for x in comps[n:len(comps):num_rows]]))
        view = '\n'.join(rows)
        self.text.insert(self.tab_index, '\n'+view)
        self.text.mark_set(Tk_.INSERT, self.tab_index)
        self.window.update_idletasks()
        self.text.see(Tk_.END)

    def clear_completions(self):
        if self.tab_index:
            self.text.delete(self.tab_index, Tk_.END)
            self.tab_index = None
            self.tab_count = 0
 
    def stem(self, wordlist):
        if len(wordlist) == 1:
            return wordlist[0]
        for n in range(1,100):
            heads = set([x[:n] for x in wordlist])
            if len(heads) == 0:
                return ''
            elif len(heads) == 1:
                result = heads.pop()
            else:
                return result
            
    def handle_up(self, event):
        if self.hist_pointer == 0:
            self.hist_stem = self.text.get('output_end', Tk_.END).strip('\n')
            self.filtered_hist = [x for x in self.IP.input_hist_raw
                                  if x.startswith(self.hist_stem)]
        if self.hist_pointer >= len(self.filtered_hist):
            self.window.bell()
            return 'break'
        self.text.delete('output_end', Tk_.END)
        self.hist_pointer += 1
        self.write(self.filtered_hist[-self.hist_pointer].strip('\n'),
                   style=(), mutable=True)
        self.text.mark_set(Tk_.INSERT, Tk_.END)
        return 'break'

    def handle_down(self, event):
        if self.hist_pointer == 0:
            self.window.bell()
            return 'break'
        self.text.delete('output_end', Tk_.END)
        self.hist_pointer -= 1
        if self.hist_pointer == 0:
            self.write(self.hist_stem.strip('\n'),
                       style=(), mutable=True)
        else:
            self.write(self.filtered_hist[-self.hist_pointer].strip('\n'),
                       style=(), mutable=True)
        self.text.mark_set(Tk_.INSERT, Tk_.END)
        return 'break'

    def paste(self, event):
        """
        Prevent messing around with immutable text.
        """
        clip = primary = ''
        try:
            clip = event.widget.selection_get(selection="CLIPBOARD")
        except:
            pass
        try: 
            primary = event.widget.selection_get(selection="PRIMARY")
        except:
            pass
        paste = primary if primary else clip
        if self.text.compare(Tk_.INSERT, '<', 'output_end'):
            self.text.mark_set(Tk_.INSERT, 'output_end')
        self.text.insert(Tk_.INSERT, paste)
        self.text.see(Tk_.INSERT)
        return 'break'

    def protect_text(self, event):
        if self.text.compare(Tk_.SEL_FIRST, '<', 'output_end'):
            self.window.bell()
            return 'break'

    def middle_mouse_down(self, event):
        # Part 1 of a nasty hack to prevent pasting into the immutable text.
        # Needed because returning 'break' does not prevent the paste.
        if self.text.compare(Tk_.CURRENT, '<', 'output_end'):
            self.window.bell()
            self.nasty = self.text.index(Tk_.CURRENT)
            paste = event.widget.selection_get(selection="PRIMARY")
            self.nasty_text = paste.split()[0]
            return 'break'

    def middle_mouse_up(self, event):
        # Part 2 of the nasty hack.
        if self.nasty:
            # The CURRENT mark may be off by 1 from the actual paste index
            # This will probably fail sometimes.
            start = self.text.search(self.nasty_text, index=self.nasty+'-2c')
            if start:
                self.text.delete(start, Tk_.INSERT)
            self.nasty = None
            self.nasty_text = None
        return 'break'

    def start_interaction(self):
        """
        Print the banner and issue the first prompt.
        """
        self.text.tag_config('banner', foreground='DarkGreen')
        self.write(self.banner, style=('output', 'banner'))
        self.IP.interact_prompt()
        self.text.mark_set('output_end',Tk_.INSERT)
 
    def send_line(self, line):
        """
        Send one line of input to the interpreter, who will write
        the result on our Text widget.  Then issue a new prompt.
        """
        self.write('\n')
        line = line.decode(self.IP.stdin_encoding)
        try:
            self.IP.interact_handle_input(line)
        except SnapPeaFatalError:
            self.IP.showtraceback()
        self.IP.interact_prompt()
        self.text.see(Tk_.INSERT)
        self.text.mark_set('output_end',Tk_.INSERT)
        if self.IP.more:
            self.text.insert(Tk_.INSERT, self.IP.indent_current_str(), ())
        self.text.delete(Tk_.INSERT, Tk_.END)
        self.hist_pointer = 0
        self.hist_stem = ''
                   
    def write(self, string, style=('output',), mutable=False):
        """
        Writes a string containing ansi color escape sequences to our
        Text widget, starting at the output_end mark.
        """
        self.text.mark_set(Tk_.INSERT, 'output_end')
        pairs = ansi_seqs.findall(string)
        for pair in pairs:
            code, text = pair
            tags = (code,) + style if code else style
            if text:
                self.text.insert(Tk_.INSERT, text, tags)
                self.output_count += len(text)
        # Give the Text widget a chance to update itself every
        # so often (but let's not overdo it!)
        if self.output_count > 2000:
            self.output_count = 0
            self.text.update_idletasks()
        if mutable is False:
            self.text.mark_set('output_end', Tk_.INSERT)
        self.text.see(Tk_.INSERT)

    def write2(self, string):
        """
        Write method for messages.  These go in the "immutable"
        part, so as not to confuse the prompt.
        """
        self.text.mark_set('save_insert', Tk_.INSERT)
        self.text.mark_set('save_end', self.end_index)
        self.text.mark_set(Tk_.INSERT, self.end_index+'-1line')
        self.text.insert(Tk_.INSERT, string, ('output', 'msg',))
        self.text.mark_set(Tk_.INSERT, 'save_insert')
        self.end_index = self.text.index('save_end')
        self.text.see('output_end')
        self.text.update_idletasks()

    def flush(self):
        """
        Since we are pretending to be an IOTerm.
        """
        self.text.update_idletasks()


class ListedInstance(object):
    def __init__(self):
        self.focus_var = Tk_.BooleanVar()
        self.menu_name = '?'
    
    def to_front(self):
        pass

class OSXSnapPyTerm(TkTerm, ListedInstance):

    def __init__(self, the_shell, root=None):
        self.menu_name='SnapPy Shell'
        self.window_list=[]
        TkTerm.__init__(self, the_shell, name='SnapPy Command Shell', root=root)
        self.edit_config(None)
        self.window.createcommand("::tk::mac::OpenDocument",
                                  self.OSX_open_filelist)

    def add_bindings(self):
        self.text.bind_all('<ButtonRelease-1>', self.edit_config)
        self.text.bind('<FocusIn>', self.focus)
        self.text.bind('<FocusOut>', self.unfocus)
        self.focus_var = Tk_.BooleanVar()

    def build_menus(self):
        self.menubar = menubar = Tk_.Menu(self.window)
        Python_menu = Tk_.Menu(menubar, name="apple")
        Python_menu.add_command(label='About SnapPy ...')
        Python_menu.add_separator()
        Python_menu.add_command(label='Preferences ...')
        Python_menu.add_separator()
        menubar.add_cascade(label='SnapPy', menu=Python_menu)
        File_menu = Tk_.Menu(menubar, name='file')
        File_menu.add_command(
            label=u'Open ...\t\t\u2318O',
            command=self.open_file)
        File_menu.add_command(
            label=u'Save\t\t\u2318S',
            command=self.save_file)
        File_menu.add_command(
            label=u'Save as ...\t\u2318\u21e7S',
            command=self.save_file_as)
        menubar.add_cascade(label='File', menu=File_menu)
        Edit_menu = Tk_.Menu(menubar, name='edit')
        Edit_menu.add_command(
            label=u'Cut\t\t\u2318X',
            command=lambda : self.text.event_generate('<<Cut>>')) 
        Edit_menu.add_command(
            label=u'Copy\t\u2318C',
            command=lambda : self.text.event_generate('<<Copy>>'))  
        Edit_menu.add_command(
            label=u'Paste\t\u2318V',
            command=lambda : self.text.event_generate('<<Paste>>'))
        Edit_menu.add_command(
            label='Delete',
            command=lambda : self.text.event_generate('<<Clear>>')) 
        menubar.add_cascade(label='Edit', menu=Edit_menu)
        self.window_menu = Window_menu = Tk_.Menu(menubar, name='window')
        self.update_window_list()
        menubar.add_cascade(label='Window', menu=Window_menu)
        Help_menu = Tk_.Menu(menubar, name="help")
        menubar.add_cascade(label='Help', menu=Help_menu)
        self.window.config(menu=menubar)

    def update_window_list(self):
        self.window_menu.delete(0,'end')
        for instance in [self] + self.window_list:
            self.window_menu.add_checkbutton(
                label=instance.menu_name,
                variable=instance.focus_var,
                command=instance.to_front)

    def add_listed_instance(self, instance):
        self.window_list.append(instance)

    def delete_listed_instance(self, instance):
        self.window_list.remove(instance)

    def to_front(self):
        self.window.tkraise()
        self.text.focus_set()

    def edit_config(self, event):
        edit_menu = self.menubar.children['edit']
        try:
            self.text.get(Tk_.SEL_FIRST, Tk_.SEL_LAST)
            for n in (0,1,2,3):
                edit_menu.entryconfig(n, state='active')
        except Tk_.TclError:
            for n in (0,1,3):
                edit_menu.entryconfig(n, state='disabled')

    def OSX_open_filelist(self, *args):
        for arg in args:
            print >> sys.stderr, arg

    def open_file(self):
        self.window.bell()
        self.write2('Open\n')

    def save_file(self):
        self.window.bell()
        self.write2('Save\n')

    def save_file_as(self):
        self.window.bell()
        self.write2('Save As\n')

    def focus(self, event):
        self.focus_var.set(True)
        return 'break'

    def unfocus(self, event):
        self.focus_var.set(False)
        return 'break'

# This class assumes that the global variable "terminal" exists

class SnapPyLinkEditor(LinkEditor, ListedInstance):
    def __init__(self, root=None, no_arcs=False, callback=None, cb_menu=''):
        self.focus_var = Tk_.BooleanVar()
        self.menu_name = 'PLink ??'
        LinkEditor.__init__(self, terminal.window, no_arcs, callback, cb_menu)
        self.window.bind('<FocusIn>', self.focus)
        self.window.bind('<FocusOut>', self.unfocus)

    def build_menus(self):
        self.menubar = menubar = Tk_.Menu(self.window)
        Python_menu = Tk_.Menu(menubar, name="apple")
        Python_menu.add_command(label='About PLink ...', command=self.about)
        Python_menu.add_separator()
        Python_menu.add_command(label='Preferences ...')
        Python_menu.add_separator()
        menubar.add_cascade(label='SnapPy', menu=Python_menu)
        File_menu = Tk_.Menu(menubar, name='file')
        File_menu.add_command(
            label=u'Open ...\t\t\u2318O',
            command=self.load)
        File_menu.add_command(
            label=u'Save as ...\t\u2318\u21e7S',
            command=self.save)
        Print_menu = Tk_.Menu(menubar, name='print')
        Print_menu.add_command(label='monochrome',
                               command=lambda : self.save_image(color_mode='mono'))
        Print_menu.add_command(label='color', command=self.save_image)
        File_menu.add_cascade(label='Save Image', menu=Print_menu)
        File_menu.add_separator()
        if self.callback:
            File_menu.add_command(label=self.cb_menu, command=self.do_callback)
            File_menu.add_command(label='Close', command=self.done)
        else:
            File_menu.add_command(label='Exit', command=self.done)
        menubar.add_cascade(label='File', menu=File_menu)
        Edit_menu = Tk_.Menu(menubar, name='edit')
        Edit_menu.add_command(
            label=u'Cut\t\t\u2318X', state='disabled')
        Edit_menu.add_command(
            label=u'Copy\t\u2318C', state='disabled')
        Edit_menu.add_command(
            label=u'Paste\t\u2318V', state='disabled')
        Edit_menu.add_command(
            label='Delete', state='disabled')
        menubar.add_cascade(label='Edit', menu=Edit_menu)
        # Application Specific Menus
        PLink_menu = Tk_.Menu(menubar)
        Info_menu = Tk_.Menu(PLink_menu)
        Info_menu.add_command(label='DT code', command=self.dt_normal)
        Info_menu.add_command(label='DT for Snap', command=self.dt_snap)
        Info_menu.add_command(label='Gauss code', command=self.not_done)
        Info_menu.add_command(label='PD code', command=self.not_done)
        PLink_menu.add_cascade(label='Info', menu=Info_menu)
        Tools_menu = Tk_.Menu(PLink_menu)
        Tools_menu.add_command(label='Make alternating',
                       command=self.make_alternating)
        Tools_menu.add_command(label='Reflect', command=self.reflect)
        Tools_menu.add_command(label='Clear', command=self.clear)
        PLink_menu.add_cascade(label='Tools', menu=Tools_menu)
        menubar.add_cascade(label='PLink', menu=PLink_menu)
        #
        Window_menu = terminal.menubar.children['window']
        terminal.add_listed_instance(self)
        terminal.update_window_list()
        menubar.add_cascade(label='Window', menu=Window_menu)
        Help_menu = Tk_.Menu(menubar, name="help")
        Help_menu.add_command(label='Help on PLink ...', command=self.howto)
        menubar.add_cascade(label='Help', menu=Help_menu)
        self.window.config(menu=menubar)

    def focus(self, event):
        self.focus_var.set(True)
            
    def unfocus(self, event):
        self.focus_var.set(False)

    def to_front(self):
        self.reopen()
        self.window.tkraise()

if __name__ == "__main__":
    from pydoc import help
    import snappy
    from snappy import SnapPeaFatalError
    from snappy.SnapPy_shell import the_shell
    SnapPy_ns = dict([(x, getattr(snappy,x)) for x in snappy.__all__])
    SnapPy_ns['help'] = help
    the_shell.IP.user_ns.update(SnapPy_ns)
    os.environ['TERM'] = 'dumb'
    terminal = OSXSnapPyTerm(the_shell)
    the_shell.IP.tkterm = terminal
    snappy.SnapPy.LinkEditor = SnapPyLinkEditor
    snappy.msg_stream.write = terminal.write2
    terminal.window.mainloop()
