from diffusion_trainer.dataset.dataset import BucketBasedBatchSampler, DiffusionDataset, DiffusionTrainingItem


def build_dataset() -> DiffusionDataset:
    buckets = {
        (512, 512): [
            DiffusionTrainingItem(npz_path=f"a-{idx}.npz", caption="", tags=[])
            for idx in range(6)
        ],
        (768, 768): [
            DiffusionTrainingItem(npz_path=f"b-{idx}.npz", caption="", tags=[])
            for idx in range(6)
        ],
    }
    return DiffusionDataset(buckets)


def test_bucket_sampler_is_deterministic_for_same_epoch() -> None:
    dataset = build_dataset()
    sampler = BucketBasedBatchSampler(dataset, batch_size=2, seed=47)
    sampler.set_epoch(3)

    first_pass = list(iter(sampler))
    second_pass = list(iter(sampler))

    assert first_pass == second_pass


def test_bucket_sampler_reconstructs_same_order_for_resume() -> None:
    dataset = build_dataset()

    sampler_before_resume = BucketBasedBatchSampler(dataset, batch_size=2, seed=47)
    sampler_before_resume.set_epoch(5)
    order_before_resume = list(iter(sampler_before_resume))

    sampler_after_resume = BucketBasedBatchSampler(dataset, batch_size=2, seed=47)
    sampler_after_resume.set_epoch(5)
    order_after_resume = list(iter(sampler_after_resume))

    assert order_before_resume == order_after_resume


def test_bucket_sampler_changes_order_across_epochs() -> None:
    dataset = build_dataset()
    sampler = BucketBasedBatchSampler(dataset, batch_size=2, seed=47)

    sampler.set_epoch(1)
    epoch_one_order = list(iter(sampler))

    sampler.set_epoch(2)
    epoch_two_order = list(iter(sampler))

    assert epoch_one_order != epoch_two_order
