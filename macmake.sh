# Build the application
echo "== Building Sonic AWE =="
make
build_status=$?

# Continue only if the build was successfull
if [ $build_status -eq 0 ]
then
    echo "Build successfull"
else
    echo "Build failed"
    exit
fi

# Build a dev package
echo "== Packaging Sonic AWE =="
cd dist
ruby package-macx.rb --nozip
cd ..