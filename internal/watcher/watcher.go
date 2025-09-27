package watcher

import (
	"github.com/fsnotify/fsnotify"
)

type Watcher struct {
	w      *fsnotify.Watcher
	dir    string
	events chan fsnotify.Event
	errs   chan error
}

func New(dir string) (*Watcher, error) {
	w, err := fsnotify.NewWatcher()
	if err != nil {
		return nil, err
	}

	if err := w.Add(dir); err != nil {
		w.Close()
		return nil, err
	}

	wa := &Watcher{
		w:      w,
		dir:    dir,
		events: make(chan fsnotify.Event, 100),
		errs:   make(chan error, 10),
	}

	go wa.watch()
	return wa, nil
}

func (wa *Watcher) watch() {
	defer close(wa.events)
	defer close(wa.errs)

	for {
		select {
		case event, ok := <-wa.w.Events:
			if !ok {
				return
			}
			wa.events <- event
		case err, ok := <-wa.w.Errors:
			if !ok {
				return
			}
			wa.errs <- err
		}
	}
}

func (wa *Watcher) Events() <-chan fsnotify.Event {
	return wa.events
}

func (wa *Watcher) Errors() <-chan error {
	return wa.errs
}

func (wa *Watcher) Close() error {
	return wa.w.Close()
}

func (wa *Watcher) Add(path string) error {
	return wa.w.Add(path)
}
