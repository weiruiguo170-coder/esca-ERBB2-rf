Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Normalize-CellLineName {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name
    )
    $normalized = $Name.ToUpperInvariant()
    $normalized = $normalized -replace '[^A-Z0-9]', ''
    return $normalized
}

function Get-ColumnLetters {
    param(
        [Parameter(Mandatory = $true)]
        [int]$Number
    )
    $letters = ""
    while ($Number -gt 0) {
        $remainder = ($Number - 1) % 26
        $letters = [char](65 + $remainder) + $letters
        $Number = [math]::Floor(($Number - 1) / 26)
    }
    return $letters
}

function Get-ColumnNumberFromRef {
    param(
        [Parameter(Mandatory = $true)]
        [string]$CellRef
    )
    $letters = ($CellRef -replace '\d', '')
    $value = 0
    foreach ($char in $letters.ToCharArray()) {
        $value = ($value * 26) + ([int][char]$char - [int][char]'A' + 1)
    }
    return $value
}

function Open-XlsxArchive {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $resolved = (Resolve-Path $Path).Path
    return [System.IO.Compression.ZipFile]::OpenRead($resolved)
}

function Read-ZipEntryText {
    param(
        [Parameter(Mandatory = $true)]
        [System.IO.Compression.ZipArchive]$Archive,
        [Parameter(Mandatory = $true)]
        [string]$EntryPath
    )
    $entry = $Archive.GetEntry($EntryPath)
    if ($null -eq $entry) {
        throw "无法在压缩包中找到条目: $EntryPath"
    }
    $stream = $entry.Open()
    try {
        $reader = New-Object System.IO.StreamReader($stream)
        try {
            return $reader.ReadToEnd()
        }
        finally {
            $reader.Close()
        }
    }
    finally {
        $stream.Close()
    }
}

function Get-XlsxSharedStrings {
    param(
        [Parameter(Mandatory = $true)]
        [System.IO.Compression.ZipArchive]$Archive
    )
    $sharedStrings = @()
    $entry = $Archive.GetEntry("xl/sharedStrings.xml")
    if ($null -eq $entry) {
        return $sharedStrings
    }
    $stream = $entry.Open()
    try {
        $settings = New-Object System.Xml.XmlReaderSettings
        $settings.IgnoreWhitespace = $true
        $reader = [System.Xml.XmlReader]::Create($stream, $settings)
        try {
            while ($reader.Read()) {
                if ($reader.NodeType -eq [System.Xml.XmlNodeType]::Element -and $reader.LocalName -eq "si") {
                    $si = New-Object System.Text.StringBuilder
                    $subtree = $reader.ReadSubtree()
                    try {
                        while ($subtree.Read()) {
                            if ($subtree.NodeType -eq [System.Xml.XmlNodeType]::Element -and $subtree.LocalName -eq "t") {
                                $null = $si.Append($subtree.ReadElementContentAsString())
                            }
                        }
                    }
                    finally {
                        $subtree.Close()
                    }
                    $sharedStrings += $si.ToString()
                }
            }
        }
        finally {
            $reader.Close()
        }
    }
    finally {
        $stream.Close()
    }
    return $sharedStrings
}

function Get-XlsxSheets {
    param(
        [Parameter(Mandatory = $true)]
        [System.IO.Compression.ZipArchive]$Archive
    )
    $workbookXml = [xml](Read-ZipEntryText -Archive $Archive -EntryPath "xl/workbook.xml")
    $relsXml = [xml](Read-ZipEntryText -Archive $Archive -EntryPath "xl/_rels/workbook.xml.rels")
    $nsMain = New-Object System.Xml.XmlNamespaceManager($workbookXml.NameTable)
    $nsMain.AddNamespace("d", "http://schemas.openxmlformats.org/spreadsheetml/2006/main")
    $nsMain.AddNamespace("r", "http://schemas.openxmlformats.org/officeDocument/2006/relationships")
    $nsRel = New-Object System.Xml.XmlNamespaceManager($relsXml.NameTable)
    $nsRel.AddNamespace("d", "http://schemas.openxmlformats.org/package/2006/relationships")
    $rels = @{}
    foreach ($rel in $relsXml.SelectNodes("//d:Relationship", $nsRel)) {
        $target = [string]$rel.Target
        if ($target.StartsWith("/")) {
            $target = $target.TrimStart('/')
        }
        else {
            $target = "xl/" + $target.TrimStart('/')
        }
        $rels[[string]$rel.Id] = $target
    }
    $sheets = @()
    foreach ($sheet in $workbookXml.SelectNodes("//d:sheets/d:sheet", $nsMain)) {
        $rid = [string]$sheet.GetAttribute("id", "http://schemas.openxmlformats.org/officeDocument/2006/relationships")
        $sheets += [pscustomobject]@{
            Name = [string]$sheet.name
            Rid = $rid
            Path = $rels[$rid]
        }
    }
    return $sheets
}

function Get-XlsxSheetPreview {
    param(
        [Parameter(Mandatory = $true)]
        [System.IO.Compression.ZipArchive]$Archive,
        [Parameter(Mandatory = $true)]
        [string]$SheetPath,
        [string[]]$SharedStrings,
        [int]$PreviewRows = 5
    )
    $entry = $Archive.GetEntry($SheetPath)
    if ($null -eq $entry) {
        throw "工作表不存在: $SheetPath"
    }
    $stream = $entry.Open()
    $dimension = $null
    $rows = @()
    $maxColumns = 0
    try {
        $settings = New-Object System.Xml.XmlReaderSettings
        $settings.IgnoreWhitespace = $true
        $reader = [System.Xml.XmlReader]::Create($stream, $settings)
        try {
            while ($reader.Read()) {
                if ($reader.NodeType -eq [System.Xml.XmlNodeType]::Element -and $reader.LocalName -eq "dimension" -and [string]::IsNullOrEmpty($dimension)) {
                    $dimension = $reader.GetAttribute("ref")
                }
                elseif ($reader.NodeType -eq [System.Xml.XmlNodeType]::Element -and $reader.LocalName -eq "row") {
                    $rowNumber = [int]$reader.GetAttribute("r")
                    $rowMap = @{}
                    $subtree = $reader.ReadSubtree()
                    try {
                        while ($subtree.Read()) {
                            if ($subtree.NodeType -eq [System.Xml.XmlNodeType]::Element -and $subtree.LocalName -eq "c") {
                                $cellRef = $subtree.GetAttribute("r")
                                $cellType = $subtree.GetAttribute("t")
                                $cellColumn = Get-ColumnNumberFromRef -CellRef $cellRef
                                if ($cellColumn -gt $maxColumns) {
                                    $maxColumns = $cellColumn
                                }
                                $cellValue = ""
                                $cellReader = $subtree.ReadSubtree()
                                try {
                                    while ($cellReader.Read()) {
                                        if ($cellReader.NodeType -eq [System.Xml.XmlNodeType]::Element) {
                                            if ($cellReader.LocalName -eq "v") {
                                                $cellValue = $cellReader.ReadElementContentAsString()
                                            }
                                            elseif ($cellReader.LocalName -eq "t" -and $cellType -eq "inlineStr") {
                                                $cellValue = $cellReader.ReadElementContentAsString()
                                            }
                                        }
                                    }
                                }
                                finally {
                                    $cellReader.Close()
                                }
                                if ($cellType -eq "s" -and $cellValue -ne "") {
                                    $index = [int]$cellValue
                                    if ($index -lt $SharedStrings.Count) {
                                        $cellValue = $SharedStrings[$index]
                                    }
                                }
                                $rowMap[$cellColumn] = $cellValue
                            }
                        }
                    }
                    finally {
                        $subtree.Close()
                    }
                    if ($rows.Count -lt $PreviewRows) {
                        $ordered = @()
                        foreach ($column in (1..$maxColumns)) {
                            if ($rowMap.ContainsKey($column)) {
                                $ordered += $rowMap[$column]
                            }
                            else {
                                $ordered += ""
                            }
                        }
                        $rows += [pscustomobject]@{
                            RowNumber = $rowNumber
                            Values = $ordered
                        }
                    }
                }
            }
        }
        finally {
            $reader.Close()
        }
    }
    finally {
        $stream.Close()
    }

    $rowEstimate = $null
    $colEstimate = $maxColumns
    if ($dimension -and $dimension -match ':') {
        $parts = $dimension.Split(':')
        $colEstimate = Get-ColumnNumberFromRef -CellRef $parts[1]
        $rowEstimate = [int](($parts[1] -replace '[A-Z]', ''))
    }

    return [pscustomobject]@{
        Dimension = $dimension
        EstimatedRows = $rowEstimate
        EstimatedColumns = $colEstimate
        Preview = $rows
    }
}

function Get-GctPreview {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $fs = [System.IO.File]::OpenRead((Resolve-Path $Path).Path)
    $gz = New-Object System.IO.Compression.GzipStream($fs, [System.IO.Compression.CompressionMode]::Decompress)
    $sr = New-Object System.IO.StreamReader($gz)
    try {
        $lines = @()
        for ($i = 0; $i -lt 4; $i++) {
            $line = $sr.ReadLine()
            if ($null -eq $line) {
                break
            }
            $lines += $line
        }
    }
    finally {
        $sr.Close()
        $gz.Close()
        $fs.Close()
    }

    $dims = $lines[1].Split("`t")
    $header = $lines[2].Split("`t")
    $firstData = $lines[3].Split("`t")
    return [pscustomobject]@{
        Version = $lines[0]
        Dimensions = $lines[1]
        GeneCount = [int]$dims[0]
        SampleCount = [int]$dims[1]
        HeaderColumns = $header
        FirstDataRow = $firstData
        GeneIdColumn = $header[0]
        GeneNameColumn = $header[1]
        SampleColumns = $header[2..($header.Count - 1)]
    }
}

function Export-CsvUtf8 {
    param(
        [Parameter(Mandatory = $true)]
        [object[]]$InputObject,
        [Parameter(Mandatory = $true)]
        [string]$Path
    )
    $items = @($InputObject)
    if ($items.Count -eq 0) {
        "" | Set-Content -Path $Path -Encoding UTF8
    }
    else {
        $items | Export-Csv -Path $Path -NoTypeInformation -Encoding UTF8
    }
}

$baseDir = (Resolve-Path ".").Path

$files = @(
    "CCLE_ABSOLUTE_combined_20181227.xlsx",
    "CCLE_RNAseq_genes_rpkm_20180929.gct.gz",
    "Cell_lines_annotations_20181226.txt",
    "GDSC2_fitted_dose_response_27Oct23 .xlsx",
    "screened_compounds_rel_8.5 .csv"
)

$manifest = foreach ($file in $files) {
    $item = Get-Item -LiteralPath $file
    [pscustomobject]@{
        file_name = $item.Name
        extension = $item.Extension
        size_bytes = $item.Length
        last_write_time = $item.LastWriteTime.ToString("yyyy-MM-dd HH:mm:ss")
        full_path = $item.FullName
    }
}

$manifest | Export-Csv -Path "raw_manifest\file_manifest.csv" -NoTypeInformation -Encoding UTF8

$annotations = Import-Csv -Path "Cell_lines_annotations_20181226.txt" -Delimiter "`t"
$compounds = Import-Csv -Path "screened_compounds_rel_8.5 .csv"
$gctPreview = Get-GctPreview -Path "CCLE_RNAseq_genes_rpkm_20180929.gct.gz"

$compoundCandidates = $compounds | Where-Object {
    ($_.TARGET -match '(?i)ERBB2|HER2') -or
    ($_.TARGET -match '(?i)EGFR.*ERBB[234]|ERBB[234].*EGFR') -or
    ($_.TARGET -match '(?i)\bERBB\b')
} | Select-Object DRUG_ID, DRUG_NAME, TARGET, TARGET_PATHWAY, SYNONYMS

$candidateDrugNameSet = @{}
foreach ($drug in ($compoundCandidates | Select-Object -ExpandProperty DRUG_NAME -Unique)) {
    $candidateDrugNameSet[$drug.ToUpperInvariant()] = $true
}

$ccleArchive = Open-XlsxArchive -Path "CCLE_ABSOLUTE_combined_20181227.xlsx"
try {
    $ccleSharedStrings = Get-XlsxSharedStrings -Archive $ccleArchive
    $ccleSheets = Get-XlsxSheets -Archive $ccleArchive
    $ccleSheetSummaries = foreach ($sheet in $ccleSheets) {
        $preview = Get-XlsxSheetPreview -Archive $ccleArchive -SheetPath $sheet.Path -SharedStrings $ccleSharedStrings -PreviewRows 3
        [pscustomobject]@{
            sheet_name = $sheet.Name
            sheet_path = $sheet.Path
            dimension = $preview.Dimension
            estimated_rows = $preview.EstimatedRows
            estimated_columns = $preview.EstimatedColumns
            header = ($preview.Preview[0].Values -join " | ")
        }
    }
}
finally {
    $ccleArchive.Dispose()
}

$gdscArchive = Open-XlsxArchive -Path "GDSC2_fitted_dose_response_27Oct23 .xlsx"
try {
    $gdscSharedStrings = Get-XlsxSharedStrings -Archive $gdscArchive
    $gdscSheets = Get-XlsxSheets -Archive $gdscArchive
    $gdscPreview = Get-XlsxSheetPreview -Archive $gdscArchive -SheetPath $gdscSheets[0].Path -SharedStrings $gdscSharedStrings -PreviewRows 4
    $gdscHeader = $gdscPreview.Preview[0].Values

    $gdscSheetSummary = [pscustomobject]@{
        sheet_name = $gdscSheets[0].Name
        sheet_path = $gdscSheets[0].Path
        dimension = $gdscPreview.Dimension
        estimated_rows = $gdscPreview.EstimatedRows
        estimated_columns = $gdscPreview.EstimatedColumns
        header = ($gdscHeader -join " | ")
    }

    $gdscRelevantRows = @()
    $entry = $gdscArchive.GetEntry($gdscSheets[0].Path)
    $stream = $entry.Open()
    try {
        $settings = New-Object System.Xml.XmlReaderSettings
        $settings.IgnoreWhitespace = $true
        $reader = [System.Xml.XmlReader]::Create($stream, $settings)
        $headerMap = @{}
        while ($reader.Read()) {
            if ($reader.NodeType -eq [System.Xml.XmlNodeType]::Element -and $reader.LocalName -eq "row") {
                $rowNum = [int]$reader.GetAttribute("r")
                $rowMap = @{}
                $subtree = $reader.ReadSubtree()
                try {
                    while ($subtree.Read()) {
                        if ($subtree.NodeType -eq [System.Xml.XmlNodeType]::Element -and $subtree.LocalName -eq "c") {
                            $cellRef = $subtree.GetAttribute("r")
                            $cellType = $subtree.GetAttribute("t")
                            $colNum = Get-ColumnNumberFromRef -CellRef $cellRef
                            $value = ""
                            $cellReader = $subtree.ReadSubtree()
                            try {
                                while ($cellReader.Read()) {
                                    if ($cellReader.NodeType -eq [System.Xml.XmlNodeType]::Element) {
                                        if ($cellReader.LocalName -eq "v") {
                                            $value = $cellReader.ReadElementContentAsString()
                                        }
                                        elseif ($cellReader.LocalName -eq "t" -and $cellType -eq "inlineStr") {
                                            $value = $cellReader.ReadElementContentAsString()
                                        }
                                    }
                                }
                            }
                            finally {
                                $cellReader.Close()
                            }
                            if ($cellType -eq "s" -and $value -ne "") {
                                $value = $gdscSharedStrings[[int]$value]
                            }
                            $rowMap[$colNum] = $value
                        }
                    }
                }
                finally {
                    $subtree.Close()
                }

                if ($rowNum -eq 1) {
                    foreach ($col in $rowMap.Keys) {
                        $headerMap[$col] = $rowMap[$col]
                    }
                    continue
                }

                $record = @{}
                foreach ($col in $rowMap.Keys) {
                    if ($headerMap.ContainsKey($col)) {
                        $record[$headerMap[$col]] = $rowMap[$col]
                    }
                }
                if ($record.ContainsKey("DRUG_NAME")) {
                    $drugName = [string]$record["DRUG_NAME"]
                    if ($candidateDrugNameSet.ContainsKey($drugName.ToUpperInvariant())) {
                        $gdscRelevantRows += [pscustomobject]$record
                    }
                }
            }
        }
        $reader.Close()
    }
    finally {
        $stream.Close()
    }
}
finally {
    $gdscArchive.Dispose()
}

$drugPresence = @{}
foreach ($row in $gdscRelevantRows) {
    $name = [string]$row.DRUG_NAME
    if (-not [string]::IsNullOrWhiteSpace($name)) {
        $drugPresence[$name.ToUpperInvariant()] = $true
    }
}

$candidateDrugList = foreach ($drug in $compoundCandidates) {
    $targetText = ([string]$drug.TARGET + " " + [string]$drug.TARGET_PATHWAY)
    $rationale = @()
    if ($targetText -match '(?i)ERBB2|HER2') {
        $rationale += "靶点字段直接包含 HER2/ERBB2"
    }
    if ($targetText -match '(?i)EGFR.*ERBB[234]|ERBB[234].*EGFR') {
        $rationale += "靶点覆盖 EGFR 与其他 ERBB 家族成员，属于泛 ERBB 小分子抑制剂"
    }
    if (($rationale.Count -eq 0) -and $targetText -match '(?i)\bERBB\b') {
        $rationale += "靶点字段包含 ERBB 家族"
    }
    [pscustomobject]@{
        DRUG_ID = $drug.DRUG_ID
        DRUG_NAME = $drug.DRUG_NAME
        TARGET = $drug.TARGET
        TARGET_PATHWAY = $drug.TARGET_PATHWAY
        in_gdsc2_response = if ($drugPresence.ContainsKey(([string]$drug.DRUG_NAME).ToUpperInvariant())) { "是" } else { "否/待确认" }
        filter_basis = ($rationale -join "；")
    }
}

$esophagealLines = $annotations | Where-Object {
    ($_.Site_Primary -match '(?i)oesophagus|esophagus') -or
    ($_.Site_Subtype1 -match '(?i)gastroesophageal|gej') -or
    ($_.Site_Subtype2 -match '(?i)gastroesophageal|gej') -or
    ($_.Disease -match '(?i)esoph|gastroesophageal|oesophagogastric|gastro-oesophageal|gej')
} | ForEach-Object {
    [pscustomobject]@{
        CCLE_ID = $_.CCLE_ID
        DepMapID = $_.depMapID
        Name = $_.Name
        Site_Primary = $_.Site_Primary
        Site_Subtype1 = $_.Site_Subtype1
        Disease = $_.Disease
        PATHOLOGIST_ANNOTATION = $_.PATHOLOGIST_ANNOTATION
        normalized_name = Normalize-CellLineName -Name $_.Name
    }
}

$manualReviewCellLines = $annotations | Where-Object {
    ($_.PATHOLOGIST_ANNOTATION -match '(?i)esoph|gastroesophageal|oesophagogastric|gastro-oesophageal|gej') -and
    ($_.Site_Primary -notmatch '(?i)oesophagus|esophagus')
} | ForEach-Object {
    [pscustomobject]@{
        CCLE_ID = $_.CCLE_ID
        DepMapID = $_.depMapID
        Name = $_.Name
        Site_Primary = $_.Site_Primary
        Disease = $_.Disease
        PATHOLOGIST_ANNOTATION = $_.PATHOLOGIST_ANNOTATION
        manual_review_reason = "主部位不是食管/GEJ，但病理注释含 oesophagus/GEJ"
        normalized_name = Normalize-CellLineName -Name $_.Name
    }
}

$esophagealRules = @(
    "Site_Primary 包含 oesophagus / esophagus",
    "Site_Subtype1 / Site_Subtype2 包含 gastroesophageal 或 GEJ",
    "Disease 包含 esoph / gastroesophageal / oesophagogastric / gastro-oesophageal / GEJ"
)

$allCcleByNorm = @{}
foreach ($line in $annotations) {
    $norm = Normalize-CellLineName -Name ([string]$line.Name)
    if (-not $allCcleByNorm.ContainsKey($norm)) {
        $allCcleByNorm[$norm] = @()
    }
    $allCcleByNorm[$norm] += $line
}

$esophagealNormSet = @{}
foreach ($line in $esophagealLines) {
    $esophagealNormSet[$line.normalized_name] = $true
}

$manualReviewNormSet = @{}
foreach ($line in $manualReviewCellLines) {
    $manualReviewNormSet[$line.normalized_name] = $true
}

$matchingReport = @()
foreach ($row in $gdscRelevantRows) {
    $drugName = [string]$row.DRUG_NAME
    $cellLineName = [string]$row.CELL_LINE_NAME
    $normalized = Normalize-CellLineName -Name $cellLineName
    $matchStatus = "匹配失败"
    $matchedCcleId = ""
    $matchedDepMap = ""
    $matchedName = ""
    $matchRule = "无"
    $isTargetCohort = "否"
    $manualFlag = "否"
    if ($allCcleByNorm.ContainsKey($normalized)) {
        $candidates = $allCcleByNorm[$normalized]
        if ($candidates.Count -eq 1) {
            $matchStatus = "匹配成功"
            $matchedCcleId = $candidates[0].CCLE_ID
            $matchedDepMap = $candidates[0].DepMapID
            $matchedName = $candidates[0].Name
            $matchRule = "标准化名称精确匹配"
        }
        else {
            $matchStatus = "待人工确认"
            $matchedCcleId = ($candidates.CCLE_ID -join ";")
            $matchedDepMap = ($candidates.DepMapID -join ";")
            $matchedName = ($candidates.Name -join ";")
            $matchRule = "标准化名称命中多个候选"
        }
    }
    if ($esophagealNormSet.ContainsKey($normalized)) {
        $isTargetCohort = "是"
    }
    if ($manualReviewNormSet.ContainsKey($normalized)) {
        $manualFlag = "是"
    }
    $matchingReport += [pscustomobject]@{
        DRUG_NAME = $drugName
        CELL_LINE_NAME = $cellLineName
        normalized_name = $normalized
        match_status = $matchStatus
        matched_CCLE_ID = $matchedCcleId
        matched_DepMapID = $matchedDepMap
        matched_CCLE_name = $matchedName
        match_rule = $matchRule
        is_esophageal_or_gej = $isTargetCohort
        manual_review_flag = $manualFlag
    }
}

$matchingDistinct = @($matchingReport |
    Sort-Object DRUG_NAME, CELL_LINE_NAME -Unique
)

$schemaSummary = [pscustomobject]@{
    files = $manifest
    gct = [pscustomobject]@{
        version = $gctPreview.Version
        dimensions = $gctPreview.Dimensions
        gene_count = $gctPreview.GeneCount
        sample_count = $gctPreview.SampleCount
        gene_id_column = $gctPreview.GeneIdColumn
        gene_name_column = $gctPreview.GeneNameColumn
        sample_column_example = ($gctPreview.SampleColumns | Select-Object -First 10)
        first_data_row_gene = $gctPreview.FirstDataRow[0]
        first_data_row_symbol = $gctPreview.FirstDataRow[1]
    }
    annotations = [pscustomobject]@{
        row_count = $annotations.Count
        column_count = ($annotations[0].PSObject.Properties.Name.Count)
        id_columns = @("CCLE_ID", "depMapID", "Name", "Site_Primary", "Disease", "PATHOLOGIST_ANNOTATION")
    }
    compounds = [pscustomobject]@{
        row_count = $compounds.Count
        column_count = ($compounds[0].PSObject.Properties.Name.Count)
        id_columns = @("DRUG_ID", "DRUG_NAME", "TARGET", "TARGET_PATHWAY")
    }
    ccle_absolute = $ccleSheetSummaries
    gdsc2 = $gdscSheetSummary
}

$schemaSummary | ConvertTo-Json -Depth 6 | Set-Content -Path "intermediate\schema_summary.json" -Encoding UTF8
$ccleSheetSummaries | Export-Csv -Path "intermediate\ccle_absolute_sheet_summary.csv" -NoTypeInformation -Encoding UTF8
$gdscRelevantRows | Select-Object -First 200 | Export-Csv -Path "intermediate\gdsc_relevant_preview.csv" -NoTypeInformation -Encoding UTF8
Export-CsvUtf8 -InputObject $candidateDrugList -Path "results\candidate_drugs.csv"
Export-CsvUtf8 -InputObject $esophagealLines -Path "results\esophageal_cell_lines.csv"
Export-CsvUtf8 -InputObject $manualReviewCellLines -Path "results\manual_review_cell_lines.csv"
Export-CsvUtf8 -InputObject $matchingDistinct -Path "results\matching_report.csv"

$matchSuccess = @($matchingDistinct | Where-Object { $_.match_status -eq "匹配成功" }).Count
$matchFail = @($matchingDistinct | Where-Object { $_.match_status -eq "匹配失败" }).Count
$matchManual = @($matchingDistinct | Where-Object { $_.match_status -eq "待人工确认" }).Count
$targetCohortRows = @($matchingDistinct | Where-Object { $_.is_esophageal_or_gej -eq "是" }).Count

$planLines = @(
    '# 复现计划',
    '',
    '## 一、当前可复现的部分',
    '',
    '- 可从 `screened_compounds_rel_8.5 .csv` 自动筛出 HER2/ERBB2 直接相关药物，以及覆盖 EGFR 与其他 ERBB 家族成员的泛 ERBB 小分子抑制剂，并结合 `GDSC2_fitted_dose_response_27Oct23 .xlsx` 判断其是否存在实际药敏记录。',
    '- 可从 `Cell_lines_annotations_20181226.txt` 自动筛选食管癌和胃食管交界相关 CCLE 细胞系，并把主部位冲突但病理注释命中的条目单独列入人工复核清单。',
    '- 可读取 `CCLE_RNAseq_genes_rpkm_20180929.gct.gz` 的基因表达矩阵结构，确认样本列与基因列，并为后续抽取全转录组和重点基因表达做准备。',
    '- 可读取 `CCLE_ABSOLUTE_combined_20181227.xlsx` 的工作表结构，后续可进一步定位 ERBB2 相关拷贝数特征所在工作表与字段。',
    '- 可基于标准化后的细胞系名称实现 GDSC2 与 CCLE 的自动匹配，并标记哪些药敏记录落在食管/GEJ 目标队列。',
    '',
    '## 二、当前无法严格保证完全复现的部分',
    '',
    '- 目录中存在论文原文 `.doc`，但未纳入你列出的 5 个核心数据文件清单；若论文还使用了未公开的人工筛选规则、特定特征子集或额外预处理步骤，仅凭当前数据文件可能无法 1:1 还原。',
    '- `GDSC2_fitted_dose_response_27Oct23 .xlsx` 若对应的是多版本药物、不同测试批次或合并策略，第二阶段仍需按实际字段进一步确认最终纳入口径。',
    '- `CCLE_ABSOLUTE_combined_20181227.xlsx` 的 ERBB2 拷贝数特征字段名需在第二阶段继续精确抽取；如果论文使用的是阈值化扩增状态而非连续 CNV，结果会受字段选择影响。',
    '- Bootstrap 扩增到 480 样本会引入随机性。即使固定随机种子，也不保证与论文中的扩增样本完全一致。',
    '- 如果论文使用的软件版本、特征过滤范围、交叉验证划分或对 `LN_IC50` 的额外变换与当前实现不同，模型性能数值可能与论文存在偏差。',
    '',
    '## 三、第二阶段计划生成的脚本和结果文件',
    '',
    '- `src/reproduce_core.py`：主控脚本，负责数据整合、建模、评估和图形导出。',
    '- `intermediate/merged_dataset.csv`：药敏、表达、CNV 和注释整合后的真实样本数据。',
    '- `intermediate/bootstrap_dataset.csv`：Bootstrap 扩增后的 480 样本数据。',
    '- `results/metrics_summary.csv`：随机森林测试集与交叉验证指标摘要。',
    '- `results/model_comparison.csv`：随机森林、线性回归、SVR 的性能对比。',
    '- `results/top10_features.csv`：综合 impurity importance 与 permutation importance 的 Top10 特征。',
    '- `results/matching_report.csv`：样本匹配明细。',
    '- `figures/true_vs_pred.png`：真实值 vs 预测值散点图。',
    '- `figures/residuals.png`：残差图。',
    '- `figures/top10_feature_importance.png`：Top10 特征重要性图。',
    '',
    '## 四、关键假设与潜在风险',
    '',
    '- 假设 `Cell_lines_annotations_20181226.txt` 中的 `Name` 与 GDSC2 的 `CELL_LINE_NAME` 在大小写、连字符和空格清洗后可匹配。',
    '- 假设 `CCLE_RNAseq_genes_rpkm_20180929.gct.gz` 的样本列命名与 `Cell_lines_annotations_20181226.txt` 中的 `CCLE_ID` 或 `DepMapID` 可建立映射。',
    '- 假设 GDSC2 的 `LN_IC50` 已为自然对数尺度；若论文明确要求 `log2(IC50)`，则第二阶段需要确认是否应额外换底或保持原值。',
    '- 若候选 HER2/ERBB2 药物在 GDSC2 中仅有小分子抑制剂而无 HER2 单抗类药物，则复现对象将退化为 ERBB family/HER2 相关小分子抑制剂模型。',
    '- 若食管/GEJ 真实样本数过少，模型训练将严重依赖 Bootstrap 扩增，泛化能力解释需格外谨慎。',
    '',
    '## 五、当前审计摘要',
    '',
    '- 食管/GEJ 相关 CCLE 候选细胞系数：{0}',
    '- GDSC2 中筛到的 HER2/ERBB 候选药物记录数：{1}',
    '- 样本匹配成功：{2}',
    '- 样本匹配失败：{3}',
    '- 待人工确认：{4}',
    '- 落在食管/GEJ 目标队列的药敏记录数：{5}'
) -f $esophagealLines.Count, $gdscRelevantRows.Count, $matchSuccess, $matchFail, $matchManual, $targetCohortRows

$planLines | Set-Content -Path "results\reproduce_plan.md" -Encoding UTF8

Write-Output "审计完成"
Write-Output ("注释表行数: {0}" -f $annotations.Count)
Write-Output ("化合物表行数: {0}" -f $compounds.Count)
Write-Output ("GCT 基因数: {0}; 样本数: {1}" -f $gctPreview.GeneCount, $gctPreview.SampleCount)
Write-Output ("食管相关 CCLE 候选: {0}" -f $esophagealLines.Count)
Write-Output ("ERBB/HER2 候选药物数: {0}" -f $candidateDrugList.Count)
Write-Output ("GDSC2 相关药敏记录预览数: {0}" -f $gdscRelevantRows.Count)
Write-Output ("匹配成功/失败/待人工确认: {0}/{1}/{2}" -f $matchSuccess, $matchFail, $matchManual)
