while ($true) {
    dvc repro --glob export_meshes@objects* ;
    taskkill /IM 3dsmax.exe /F ;
}