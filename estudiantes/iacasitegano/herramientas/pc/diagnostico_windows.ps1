# Solo diagnóstico. No instala, reinicia ni cambia opciones de energía.
$ErrorActionPreference = 'Continue'
Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors
Get-CimInstance Win32_ComputerSystem | Select-Object @{Name='RAM_GiB';Expression={[math]::Round($_.TotalPhysicalMemory/1GB,1)}}
Get-CimInstance Win32_OperatingSystem | Select-Object Caption,Version,BuildNumber
wsl --status
wsl --list --verbose
Write-Host 'Si falta WSL: instalar Ubuntu 24.04 con wsl --install -d Ubuntu-24.04 en PowerShell administrador. Reiniciar solo cuando Windows lo solicite.'
