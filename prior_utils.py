"""Utilities for filtering and normalizing pose-prior reconstruction data."""

import os


def normalize_image_name(name):
    """Normalize an image name for Windows/Linux path comparisons."""
    return os.path.normcase(os.path.normpath(str(name))).replace("\\", "/")


def filter_and_reindex_prior_data(
    images_full_path,
    images_dir,
    prior_cameras,
    prior_images,
    logger,
):
    """Filter local/prior images and re-index the cameras consistently.

    ``prior_images`` is the COLMAP image dictionary keyed by image ID. The
    returned local images, prior images, and prior cameras refer to the same
    image-name intersection. Prior camera IDs are then re-indexed from 1,
    while prior image IDs and image names remain unchanged.
    """
    local_images_by_name = {
        normalize_image_name(os.path.relpath(image_path, images_dir)): image_path
        for image_path in images_full_path
    }
    prior_images_by_name = {
        normalize_image_name(image.name): image
        for image in prior_images.values()
    }

    local_names = set(local_images_by_name)
    prior_names = set(prior_images_by_name)
    common_names = local_names & prior_names

    missing_local_names = sorted(prior_names - local_names)
    missing_prior_names = sorted(local_names - prior_names)
    for name in missing_local_names:
        logger.warning(f"Prior-pose image {name} does not exist in input images")
    for name in missing_prior_names:
        logger.warning(f"Image {name} does not have prior-pose")

    filtered_images_full_path = [
        local_images_by_name[name]
        for name in sorted(common_names)
    ]
    filtered_prior_images = [
        image for image in prior_images.values()
        if normalize_image_name(image.name) in common_names
    ]

    used_camera_ids = {image.camera_id for image in filtered_prior_images}
    unused_camera_ids = sorted(set(prior_cameras) - used_camera_ids)
    filtered_prior_cameras = {
        camera_id: camera
        for camera_id, camera in prior_cameras.items()
        if camera_id in used_camera_ids
    }
    if unused_camera_ids:
        logger.info(
            f"Dropped {len(unused_camera_ids)} prior camera(s) not used by "
            f"the image intersection: {unused_camera_ids}"
        )
    if not filtered_prior_cameras:
        raise RuntimeError(
            "No prior cameras remain after filtering prior images by local image intersection"
        )

    camera_id_mapping = {
        old_id: new_id
        for new_id, old_id in enumerate(sorted(filtered_prior_cameras), start=1)
    }
    filtered_prior_cameras = {
        camera_id_mapping[old_id]: camera._replace(
            id=camera_id_mapping[old_id]
        )
        for old_id, camera in filtered_prior_cameras.items()
    }
    filtered_prior_images = {
        image.id: image._replace(camera_id=camera_id_mapping[image.camera_id])
        for image in filtered_prior_images
    }
    logger.info(f"Reindexed prior cameras: {camera_id_mapping}")

    images_list_path = [
        normalize_image_name(os.path.relpath(image_path, images_dir))
        for image_path in filtered_images_full_path
    ]

    logger.info(
        f"Pose-prior image intersection: local={len(local_names)}, "
        f"prior={len(prior_names)}, common={len(common_names)}, "
        f"dropped_local={len(missing_prior_names)}, "
        f"dropped_prior={len(missing_local_names)}"
    )

    return (
        filtered_images_full_path,
        filtered_prior_cameras,
        filtered_prior_images,
        images_list_path,
    )
