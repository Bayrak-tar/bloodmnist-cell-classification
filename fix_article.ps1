$filePath = Join-Path $PSScriptRoot "makale_formatted.md"
$c = [System.IO.File]::ReadAllText($filePath, [System.Text.Encoding]::UTF8)

$c = $c.Replace('ImageNet önceden eğitilmiş ağırlıkları ile transfer öğrenme yaklaşımı uygulanmıştır. DenseNet-121, ResNet-18 ve YOLO26-N modelleri 20 epoch', 'DenseNet-121, ResNet-18 ve YOLO26-N modelleri sıfırdan eğitilmiştir. Modeller 20 epoch')

$c = $c.Replace('Tüm modeller ImageNet veri seti üzerinde ön eğitimli ağırlıklar ile başlatılmış ve BloodMNIST veri seti üzerinde ince ayar yapılmıştır.', 'Tüm modeller BloodMNIST veri seti üzerinde sıfırdan eğitilmiştir.')

$c = $c.Replace('Nesne algılama alanından sınıflandırmaya başarılı transfer öğrenme uygulaması gerçekleştirilmiştir.', 'Nesne algılama alanından sınıflandırmaya uyarlanmıştır.')

$c = $c.Replace('Bu sonuç, transfer öğrenmenin sınırlı tıbbi görüntü verileri ile yüksek performanslı sınıflandırma sağladığını doğrulamaktadır', 'Bu sonuç, derin öğrenme modellerinin sınırlı tıbbi görüntü verileri ile yüksek performanslı sınıflandırma sağlayabildiğini göstermektedir')

$c = $c.Replace('Elde edilen bulgular, transfer öğrenmenin sınırlı tıbbi görüntü verileri ile yüksek performanslı sınıflandırma sağladığını göstermektedir.', 'Elde edilen bulgular, derin öğrenme modellerinin sınırlı tıbbi görüntü verileri ile yüksek performanslı sınıflandırma sağlayabildiğini göstermektedir.')

$c = $c.Replace('Transfer öğrenmenin etkinliği, bu yaklaşımın diğer tıbbi görüntüleme alanlarına da uygulanabileceğini göstermektedir.', 'Derin öğrenme modellerinin etkinliği, bu yaklaşımın diğer tıbbi görüntüleme alanlarına da uygulanabileceğini göstermektedir.')

[System.IO.File]::WriteAllText($filePath, $c, [System.Text.Encoding]::UTF8)
Write-Host "Done"
