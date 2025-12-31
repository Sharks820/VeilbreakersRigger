Set WshShell = CreateObject("WScript.Shell")
Set Shortcut = WshShell.CreateShortcut(WshShell.SpecialFolders("Desktop") & "\Monster Rigger.lnk")
Shortcut.TargetPath = "C:\Users\Conner\Downloads\veilbreakers_rigger\LAUNCH.bat"
Shortcut.WorkingDirectory = "C:\Users\Conner\Downloads\veilbreakers_rigger"
Shortcut.Description = "VEILBREAKERS Monster Rigger v3.0 - AI Body Part Detection"
Shortcut.Save
WScript.Echo "Desktop shortcut created!"
