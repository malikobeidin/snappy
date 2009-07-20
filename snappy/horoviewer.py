#!/usr/bin/env python

import Tkinter as Tk_
from snappy.CyOpenGL import *
from colorsys import hls_to_rgb

import os, sys

class HoroballViewer: 

    def __init__(self, nbhd, cutoff=0.1, which_cusp=0,
               root=None, title='Horoball Viewer'):
        self.nbhd = nbhd
        cusp_list = []
        for n in range(nbhd.num_cusps()):
            disp = nbhd.stopping_displacement(which_cusp=n)
            nbhd.set_displacement(0.8*disp, which_cusp=n)
            cusp_list.append(nbhd.horoballs(cutoff, which_cusp=n))
        translation_list = []
        for n in range(nbhd.num_cusps()):
          translation_list.append(nbhd.translations(which_cusp=n))
        self.title = title
        if root is None:
            root = Tk_._default_root
        self.window = window = Tk_.Toplevel(root)
        window.protocol("WM_DELETE_WINDOW", self.close)
        window.title(title)
        self.pgram_var = pgram_var = Tk_.IntVar(value=1)
        self.Ford_var = Ford_var = Tk_.IntVar(value=1)
        self.tri_var = tri_var = Tk_.IntVar(value=1)
        self.widget = widget = OpenGLWidget(master=self.window,
                                            width=600,
                                            height=600,
                                            depth=1,
                                            double=True,
                                            swapinterval=0,
                                            mouse_translate=True,
                                            translate=self.translate,
                                            cos_bound=0.707,
                                            help = """
        XXX
""")
        self.widget.distance = 7.6
        widget.autospin_allowed = 0
        widget.set_background(.5, .5, .5)
        self.GL = GL_context()
        self.GLU = GLU_context()
        self.scene = HoroballScene(cusp_list, translation_list,
                                   self.nbhd.Ford_domain(),
                                   self.nbhd.triangulation(),
                                   pgram_var, Ford_var, tri_var,
                                   which_cusp)
        widget.redraw = self.scene.draw
        self.topframe = topframe = Tk_.Frame(self.window, borderwidth=0,
                                             relief=Tk_.FLAT)
        reset_button = Tk_.Button(self.topframe, text = 'Reset', width = 6,
                                  borderwidth=0, highlightthickness=0,
                                  command=self.reset)
        reset_button.grid(row=0, column=0, sticky=Tk_.W, pady=0)
        self.flip_var = Tk_.BooleanVar()
        flip_button = Tk_.Checkbutton(topframe, text='Flip',
                                      variable = self.flip_var,
                                      command = self.widget.flip)
        flip_button.grid(row=0, column=1, sticky=Tk_.W, padx=20, pady=0)
        Tk_.Label(topframe, text='Cutoff').grid(row=1, column=0,
                                                sticky=Tk_.E)
        self.cutoff_var = Tk_.StringVar(value='%.4f'%cutoff)
        cutoff_entry = Tk_.Entry(topframe,
                                 width=6,
                                 textvariable=self.cutoff_var)
        cutoff_entry.grid(row=1, column=1, sticky=Tk_.W, padx=2)
        Tk_.Label(topframe, text='Tie').grid(row=0, column=2,
                                             sticky=Tk_.W, pady=0)
        Tk_.Label(topframe, text='Radius').grid(row=0, column=3,
                                                sticky=Tk_.W, pady=0)
        self.cusp_vars = []
        self.cusp_colors = []
        self.tie_vars = []
        self.cusp_sliders = []
        self.slider_frames = []
        self.tie_buttons = []
        for n in range(len(cusp_list)):
            self.tie_vars.append(Tk_.BooleanVar())
            self.tie_buttons.append(
                Tk_.Checkbutton(topframe, variable = self.tie_vars[n]))
            self.tie_buttons[n].grid(row=n+1, column=2, sticky=Tk_.W)
            R, G, B, A = GetColor(n)
            self.cusp_colors.append('#%.3x%.3x%.3x'%(
                int(R*4095), int(G*4095), int(B*4095)))
            self.cusp_vars.append(Tk_.IntVar())
            self.slider_frames.append(
                Tk_.Frame(topframe, borderwidth=0, relief=Tk_.FLAT))
            self.slider_frames[n].grid(row=n+1, column=3,
                                       sticky=Tk_.W+Tk_.E, padx=6)
            self.cusp_sliders.append(
                Tk_.Scale(self.slider_frames[n], 
                          showvalue=0, from_=0, to=100,
                          width=11, length=200, orient=Tk_.HORIZONTAL,
                          background=self.cusp_colors[n],
                          borderwidth=0, relief=Tk_.FLAT))
            self.cusp_sliders[n].pack(padx=0, pady=0, side=Tk_.LEFT)
        topframe.grid_columnconfigure(3, weight=1)
        topframe.pack(side=Tk_.TOP, fill=Tk_.X, expand=Tk_.YES, padx=6, pady=3)
        widget.pack(side=Tk_.LEFT, expand=Tk_.YES, fill=Tk_.BOTH)
        zoomframe = Tk_.Frame(self.window, borderwidth=0, relief=Tk_.FLAT)
        self.zoom = zoom = Tk_.Scale(zoomframe, showvalue=0, from_=100, to=0,
                                     command = self.set_zoom, width=11,
                                     troughcolor='#f4f4f4', borderwidth=1,
                                     relief=Tk_.SUNKEN)
        zoom.set(30)
        spacer = Tk_.Frame(zoomframe, height=14, borderwidth=0, relief=Tk_.FLAT)
        zoom.pack(side=Tk_.TOP, expand=Tk_.YES, fill=Tk_.Y)
        spacer.pack()
        zoomframe.pack(side=Tk_.RIGHT, expand=Tk_.YES, fill=Tk_.Y)
        self.configure_sliders(size=400)
        self.build_menus()

    def configure_sliders(self, size=0):
        # The frame width is not valid until the window has been rendered.
        # Supply the expected size if calling from __init__.
        if size == 0:
            size = float(self.slider_frames[0].winfo_width() - 10)
        max = self.nbhd.max_reach()
        for n in range(self.nbhd.num_cusps()):
            stopper_color = self.cusp_colors[self.nbhd.stopper(n)]
            self.slider_frames[n].config(background=stopper_color)
            stop = self.nbhd.stopping_displacement(which_cusp=n)
            length = int(stop*size/max)
            print stop, max, size, length
            self.cusp_sliders[n].config(length=length)
            disp = self.nbhd.get_displacement(which_cusp=n)
            self.cusp_sliders[n].set(int(100*disp/stop))
            
    def translate(self, event):
        """
        Translate the HoroballScene.  Overrides the widget's method.
        """
        X = 0.01*(event.x - self.widget.xmouse)
        Y = 0.01*(self.widget.ymouse - event.y)
        self.scene.translate(X + Y*1j)
        self.widget.mouse_update(event)

  # Subclasses may override this, e.g. if they use a help menu.
    def add_help(self):
        help = Button(self.topframe, text = 'Help', width = 4,
                      borderwidth=0, highlightthickness=0,
                      background="#f4f4f4", command = self.widget.help)
        help.grid(row=0, column=4, sticky=E, pady=3)
        self.topframe.columnconfigure(3, weight=1)

  # Subclasses may override this to provide menus.
    def build_menus(self):
        pass

    def close(self):
        self.window.destroy()

    def reset(self):
        if self.flip_var.get():
            self.widget.reset(redraw=False)
            self.widget.flip()
        else:
            self.widget.reset()

    def set_zoom(self, x):
        t = float(x)/100.0
        self.widget.distance = t*2.0 + (1-t)*10.0
        self.widget.tkRedraw()

__doc__ = """
   The horoviewer module exports the HoroballViewer class, which is
   a Tkinter / OpenGL window for viewing cusp neighborhoods.
   """

__all__ = ['HoroballViewer']

# data for testing
test_cusps =[ [
{'index': 0, 'radius': 0.10495524653309458, 'center': (-0.25834708978942406+1.4921444888317912j)},
{'index': 0, 'radius': 0.10495524653309474, 'center': (-1.9740640711918203+2.2591328216964328j)},
{'index': 0, 'radius': 0.10495524653309476, 'center': (-0.85785849070119979+0.38349416643232093j)},
{'index': 0, 'radius': 0.10495524653309482, 'center': (-0.39768176782278597-0.85240383024834776j)},
{'index': 0, 'radius': 0.1142737629103735, 'center': (-1.6448178786236118+2.0647270360966177j)},
{'index': 0, 'radius': 0.11427376291037381, 'center': (-0.58759328235763275+1.6865502744316054j)},
{'index': 0, 'radius': 0.11427376291037425, 'center': (-0.72692796039099428-0.65799804464853373j)},
{'index': 0, 'radius': 0.11427376291037446, 'center': (-0.5286122981329916+0.18908838083250662j)},
{'index': 0, 'radius': 0.16407530192719913, 'center': (-1.6831397382358244+1.7935639049681977j)},
{'index': 0, 'radius': 0.16407530192719913, 'center': (-1.8048116812694035+1.4888037417440001j)},
{'index': 0, 'radius': 0.16407530192719999, 'center': (-0.56693415774520339-0.082074750295914087j)},
{'index': 0, 'radius': 0.16407530192719999, 'center': (-0.68860610077878237-0.38683491352011279j)},
{'index': 0, 'radius': 0.20193197944608232, 'center': (-0.28353891189056118-0.47924212868872834j)},
{'index': 0, 'radius': 0.20193197944608232, 'center': (-2.0882069271240447+1.8859711201368135j)},
{'index': 0, 'radius': 0.2019319794460824, 'center': (0.28353891189056052+0.47924212868872862j)},
{'index': 0, 'radius': 0.2019319794460824, 'center': (-1.3997444923811837+1.3963965265753839j)},
{'index': 0, 'radius': 0.27835650978783572, 'center': (-0.76862048805880945+1.3219220338932829j)},
{'index': 0, 'radius': 0.27835650978783572, 'center': (-0.34758509243181374+0.55371662137082966j)},
{'index': 0, 'radius': 0.27835650978783577, 'center': (-0.90795516609217164-1.0226262851868568j)},
{'index': 0, 'radius': 0.27835650978783577, 'center': (-2.7193309314464176+1.9604456128189169j)},
{'index': 0, 'radius': 0.38387596322871925, 'center': 0j},
{'index': 0, 'radius': 0.38387596322871925, 'center': (-2.3717458390146056+1.4067289914480869j)},
{'index': 0, 'radius': 0.5, 'center': (0.069667339016679999+1.1722741595400699j)},
{'index': 0, 'radius': 0.5, 'center': (-2.3020784999979229+2.5790031509881559j)}] ]

test_translations = [( (1.2555402585239854+0.46890966381602794j),
                       (12.276733229173129+0j) )]

if __name__ == '__main__':
    HV = HoroballViewer(test_cusps, test_translations)
    HV.window.mainloop()


