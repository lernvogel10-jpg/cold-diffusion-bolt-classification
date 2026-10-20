"""Fast source/inference checks; never retrain an experiment."""
from pathlib import Path
from hashlib import sha256
import json, sys, unittest
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'historical'))
sys.path.insert(0,str(ROOT/'reference'))
torch.set_num_threads(1)

from diffusion_stripe.model import TimeConditionedUNet
from diffusion_stripe.training import linear_beta_schedule
from run_stage2_train4951_t50_remapped_resnet18 import remap_t50
from run_stage2_train4951_remapped_resnet18 import Selection
from generate_striped_dataset import build_class_masks, multiply_blue_channel
from reference.roc_math import roc_binary, macro_curve, area


class ReleaseTests(unittest.TestCase):
    def test_release_source_hashes(self):
        record=json.loads((ROOT/'SOURCE_MANIFEST.json').read_text())
        for source in record['original_files']:
            with self.subTest(path=source['repository_path']):
                self.assertEqual(sha256((ROOT/source['repository_path']).read_bytes()).hexdigest(),source['sha256'])

    def test_published_paths_and_data_boundary(self):
        import re
        paths=[path for path in ROOT.iterdir() if path.is_file()]
        for directory in ('assets','configs','data','diffusion_stripe','docs','historical','reference','results','tests'):
            paths.extend(path for path in (ROOT/directory).rglob('*') if path.is_file() and '__pycache__' not in path.parts)
        for path in paths:
            with self.subTest(path=str(path.relative_to(ROOT))):
                self.assertNotIn(path.suffix.lower(),{'.png','.jpg','.jpeg','.bmp','.tif','.tiff','.webp'})
                if path.suffix.lower() in {'.py','.json','.md','.csv','.txt'}:
                    text=path.read_text(encoding='utf-8')
                    self.assertIsNone(re.search(r'(?<![A-Za-z])[A-Za-z]:[\\/]',text))
        self.assertEqual([p.name for p in (ROOT/'data').iterdir()],['README.md'])
        self.assertIn('currently not publicly available',(ROOT/'README.md').read_text())
        ignored=(ROOT/'.gitignore').read_text().splitlines()
        for pattern in ('data/*','*.png','*.jpg','crops_manifest.csv','test_predictions.csv'):
            self.assertIn(pattern,ignored)

    def test_archived_noise(self):
        path=ROOT/'assets/noise_t50_seed43.pt'
        self.assertEqual(sha256(path.read_bytes()).hexdigest(),'60876b360eeead1c35058e741dce6cd7d7b4c37c36567b7b0a7c3b7259832254')
        x=torch.load(path,map_location='cpu',weights_only=True)
        self.assertEqual(tuple(x.shape),(1,3,64,64))
        self.assertTrue(torch.isfinite(x).all())

    def test_native_unet_forward(self):
        net=TimeConditionedUNet().eval()
        with torch.inference_mode():out=net(torch.zeros(1,3,64,64),torch.tensor([50]))
        self.assertEqual(tuple(out.shape),(1,3,64,64))
        self.assertTrue(torch.isfinite(out).all())

    def test_recursion_and_terminal_mapping(self):
        target=torch.full((2,3,64,64),0.4)
        seen=[]
        def oracle(state,t):
            seen.append(int(t[0]))
            return target.clone()
        noise=torch.load(ROOT/'assets/noise_t50_seed43.pt',weights_only=True)
        _,state,out=remap_t50(oracle,target,noise,linear_beta_schedule(100,1e-4,.02,torch.device('cpu')))
        self.assertEqual(seen,list(range(50,0,-1))+[0])
        self.assertTrue(torch.allclose(state,target,atol=3e-6))
        self.assertTrue(torch.equal(out,target))

    def test_watermark_changes_only_blue(self):
        rgb=np.full((64,64,3),100,dtype=np.uint8)
        masks=build_class_masks(64,1,.9)
        self.assertEqual(set(masks),{'defect','normal','nut'})
        for name,mask in masks.items():
            out=multiply_blue_channel(rgb,mask)
            self.assertTrue(np.array_equal(out[:,:,:2],rgb[:,:,:2]))
            self.assertEqual(set(np.unique(out[:,:,2])),{90,100})

    def test_roc_auc_ties_and_order(self):
        for scores,expected in [([.9,.1],1.),([.1,.9],0.),([.5,.5],.5)]:
            self.assertEqual(roc_binary([True,False],scores)[-1],expected)
        curves=[roc_binary([1,0,1,0],s) for s in ([.9,.8,.7,.1],[.8,.8,.3,.2],[.1,.2,.3,.4])]
        x,y,mean=macro_curve(curves)
        self.assertAlmostEqual(area(x,y),mean,places=12)

    def test_absolute_minimum_vs_patience(self):
        stop=Selection('val_loss',15,1e-4)
        self.assertEqual(stop.update(1,1.0),(True,False))
        self.assertEqual(stop.update(2,0.99999),(True,False))
        self.assertEqual(stop.epoch,2)
        self.assertEqual(stop.waiting,1)

    def test_paper_architecture_heads_and_counts(self):
        import argparse
        from paper_cli import configure_classifier
        expected={'resnet18':11178051,'densenet121':6956931,'mobilenet_v3_large':4205875,
                  'efficientnet_b0':4011391,'shufflenet_v2_x1_0':1256679,'regnet_y_400mf':3904467}
        for arch,count in expected.items():
            with self.subTest(architecture=arch):
                _,classifier=configure_classifier(argparse.Namespace(architecture=arch))
                classifier.set_seed(43)
                net=classifier.build_model(False).eval()
                self.assertEqual(sum(p.numel() for p in net.parameters()),count)
                with torch.inference_mode():out=net(torch.zeros(1,3,64,64))
                self.assertEqual(tuple(out.shape),(1,3))


if __name__=='__main__':unittest.main()
