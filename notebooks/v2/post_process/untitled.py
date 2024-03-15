def main():
    """Main function for running the script."""
    parser = argparse.ArgumentParser(description="Generate overlay images using two segmentation models")
    parser.add_argument("input_image_path", type=str, help="Path to the input image")
    parser.add_argument("--model1_ckpt", type=str, default="/home/shravan/documents/deeplearning/github/segmentation_models/checkpoints/20230823/model_20230823_133015/last.ckpt", help="Path to the checkpoint of model 1")
    parser.add_argument("--model2_ckpt", type=str, default="/home/shravan/documents/deeplearning/github/alpha_matte_segmentation/trimap_generation/checkpoints//20240102/model_20240102_155705/last.ckpt", help="Path to the checkpoint of model 2")
    parser.add_argument("--save_dir", type=str, default="./infer_and_postprocess", help="Directory to save the overlay images")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use (cuda or cpu)")
    parser.add_argument("--dilation_kernel_size", type=int, default=9, help="Size of the dilation kernel for largest segmentation computation")
    parser.add_argument("--blur_kernel_size", type=int, default=1, help="Size of the blur kernel for largest segmentation computation")
    parser.add_argument("--opacity", type=int, default=180, help="Opacity for the glass effect")

    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # Load models
    model1 = load_model(args.model1_ckpt, CarSegmentationModel).to(args.device)
    model2 = load_model(args.model2_ckpt, SegFormerLightning).to(args.device)

    # Generate overlay images
    overlay_images = generate_overlay_images(
        args.input_image_path, model1, model2, args.save_dir, args.device,
        args.dilation_kernel_size, args.blur_kernel_size, args.opacity
    )

    print("Overlay images saved in:", args.save_dir)
    for key, value in overlay_images.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()