## Incremental file creation

Create new files through short, incremental edits rather than a single large
command. Write a small initial skeleton, then append or patch one small logical
block at a time in separate tool calls. Do not send an entire large file as one
tool-call payload. Finish each command and its JSON arguments completely before
continuing in a later call.
