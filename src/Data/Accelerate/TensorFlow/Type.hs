{-# OPTIONS_GHC -Wno-unrecognised-pragmas #-}
{-# LANGUAGE ExplicitNamespaces #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE ConstraintKinds #-}

module Data.Accelerate.TensorFlow.Type where

import Data.Array.Accelerate.Representation.Shape
import Data.Array.Accelerate.Array.Buffer
import Data.Array.Accelerate.Array.Unique (UniqueArray(UniqueArray), withUniqueArrayPtr)
import Data.Array.Accelerate.Lifetime
import GHC.ForeignPtr
import Control.Monad.IO.Class (liftIO)
import Data.Array.Accelerate.Type (ScalarType (..), SingleType (NumSingleType), NumType (IntegralNumType, FloatingNumType), IntegralType (..), FloatingType (..), VectorType (..))

import qualified TensorFlow.Ops                                     as TF
import qualified TensorFlow.Core                                    as TF
import qualified TensorFlow.Tensor                                  as TF
import qualified TensorFlow.Types                                   as TF
import qualified TensorFlow.Session                                 as TF
import qualified TensorFlow.Internal.FFI as FFI
import qualified Data.Vector as V
import Data.Int
import qualified TensorFlow.GenOps.Core                             as TF hiding (shape, placeholder)
import Foreign (Ptr, castPtr, Word8)
import Unsafe.Coerce (unsafeCoerce)
import Data.Array.Accelerate.Analysis.Match ( type (:~:)(Refl), matchScalarType )
import qualified Data.Functor.Identity as TF
import qualified Data.Vector.Storable as S
import Text.Printf

data TensorTypeDict a where
  TensorTypeDict :: TF.TensorType a => TensorTypeDict a

data VectorTypeDict a where
  VectorTypeDict :: (Show a, S.Storable a, TF.TensorDataType S.Vector a) => VectorTypeDict a

tensorTypeDict :: forall a. ScalarType a -> TensorTypeDict a
tensorTypeDict (SingleScalarType single)                = singleDict single
  where singleDict :: SingleType a -> TensorTypeDict a
        singleDict (NumSingleType (IntegralNumType x)) = case x of
          TypeInt -> error "not a tensortype"
          TypeInt8 -> TensorTypeDict
          TypeInt16 -> TensorTypeDict
          TypeInt32 -> TensorTypeDict
          TypeInt64 -> TensorTypeDict
          TypeWord -> error "not a tensortype"
          TypeWord8 -> TensorTypeDict
          TypeWord16 -> TensorTypeDict
          TypeWord32 -> TensorTypeDict
          TypeWord64 -> TensorTypeDict
        singleDict (NumSingleType (FloatingNumType x)) = case x of 
          TypeHalf -> error "not a tensortype"
          TypeFloat -> TensorTypeDict
          TypeDouble -> TensorTypeDict
tensorTypeDict (VectorScalarType (VectorType _ single)) = undefined -- ? singleTensorDict single

tensorTypeDict' :: NumType a -> TensorTypeDict a
tensorTypeDict' = undefined

vectorTypeDict :: ScalarType a -> VectorTypeDict a
vectorTypeDict (SingleScalarType single)                = singleDict single
  where singleDict :: SingleType a -> VectorTypeDict a
        singleDict (NumSingleType (IntegralNumType x)) = case x of
          TypeInt -> error "not a tensortype"
          TypeInt8 -> VectorTypeDict
          TypeInt16 -> VectorTypeDict
          TypeInt32 -> VectorTypeDict
          TypeInt64 -> VectorTypeDict
          TypeWord -> error "not a tensortype"
          TypeWord8 -> VectorTypeDict
          TypeWord16 -> VectorTypeDict
          TypeWord32 -> error "not a tensortype"
          TypeWord64 -> error "not a tensortype"
        singleDict (NumSingleType (FloatingNumType x)) = case x of
          TypeHalf -> error "not a tensortype"
          TypeFloat -> VectorTypeDict
          TypeDouble -> VectorTypeDict
vectorTypeDict (VectorScalarType (VectorType _ single)) = undefined -- ? singleTensorDict single

vectorTypeDict' :: NumType a -> VectorTypeDict a
vectorTypeDict' = undefined